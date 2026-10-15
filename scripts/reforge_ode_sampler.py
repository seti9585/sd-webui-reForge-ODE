"""Standalone reForge ODE sampler with independent settings and per-pass methods.

SPDX-License-Identifier: AGPL-3.0-only

The ODE core originates from redhottensors/ComfyUI-ODE (GPL version 3).
See NOTICE.md and LICENSE (Part 2).
The ODE fallback follows Panchovix/stable-diffusion-webui-reForge,
ldm_patched/k_diffusion/sampling.py (GNU AGPL version 3).
https://github.com/Panchovix/stable-diffusion-webui-reForge
"""

from __future__ import annotations

import logging

import gradio as gr
import torch

from modules import sd_samplers_common, script_callbacks, shared
from modules.shared import opts

LOG = logging.getLogger("reForge-ODE Sampler")

ADAPTIVE_SOLVERS = {"dopri8", "dopri5", "bosh3", "fehlberg2", "adaptive_heun"}
FIXED_SOLVERS = {"euler", "midpoint", "rk4", "heun3", "explicit_adams", "implicit_adams"}
SOLVERS = sorted(ADAPTIVE_SOLVERS | FIXED_SOLVERS)

USE_SAME = "Use same sampler"
LEGACY_SETTINGS = "Settings default"
LEGACY_SAME_TXT = "Same solver as txt2img"
NAME = "reForge-ODE Sampler"

KEY_SOLVER = "reforge_ode_sampler_solver"
KEY_RTOL = "reforge_ode_sampler_log_rtol"
KEY_ATOL = "reforge_ode_sampler_log_atol"
KEY_MAX = "reforge_ode_sampler_max_steps"


def _option(key, default):
    return getattr(opts, key, default)


def _valid_solver(value):
    if value in SOLVERS:
        return value
    LOG.warning("Invalid solver %r; falling back to dopri5", value)
    return "dopri5"


@torch.no_grad()
def _ode_function_replay(model, x, sigmas, extra_args=None, callback=None, disable=None,
                         solver="dopri5", rtol=0.001, atol=0.0001, max_steps=250):
    """Vendored fallback matching reForge's ODEFunction and ODESampler path."""
    import torchdiffeq
    from tqdm.auto import tqdm, trange

    class ODEFunction:
        def __init__(self, model, t_min, t_max, n_steps, is_adaptive, extra_args=None, callback=None):
            self.model = model
            self.extra_args = {} if extra_args is None else extra_args
            self.callback = callback
            self.t_min = t_min.item()
            self.t_max = t_max.item()
            self.n_steps = n_steps
            self.is_adaptive = is_adaptive
            self.step = 0
            self.pbar = tqdm(total=100 if is_adaptive else n_steps, desc="solve",
                             unit="%" if is_adaptive else "step", leave=False, position=1)

        def __call__(self, t, y):
            if t <= 0.00001:
                return torch.zeros_like(y)
            denoised = self.model(y.unsqueeze(0), t.unsqueeze(0), **self.extra_args)
            return (y - denoised.squeeze(0)) / t

        def _callback(self, t0, y0, step):
            if self.callback is not None:
                y0 = y0.unsqueeze(0)
                self.callback({"x": y0, "i": step, "sigma": t0,
                               "sigma_hat": t0, "denoised": y0})

        def callback_step(self, t0, y0, dt):
            if self.is_adaptive:
                return
            self._callback(t0, y0, self.step)
            self.pbar.update(1)
            self.step += 1

        def callback_accept_step(self, t0, y0, dt):
            if not self.is_adaptive:
                return
            progress = (self.t_max - t0.item()) / (self.t_max - self.t_min)
            self._callback(t0, y0, round((self.n_steps - 1) * progress))
            new_step = round(100 * progress)
            self.pbar.update(new_step - self.step)
            self.step = new_step

        def reset(self):
            self.step = 0
            self.pbar.reset()

    t_max = sigmas.max()
    t_min = sigmas.min()
    n_steps = len(sigmas)
    is_adaptive = solver not in FIXED_SOLVERS
    t = torch.stack([t_max, t_min]) if is_adaptive else sigmas
    ode = ODEFunction(model, t_min, t_max, n_steps, is_adaptive,
                      extra_args=extra_args, callback=callback)

    samples = torch.empty_like(x)
    try:
        for i in trange(x.shape[0], desc=solver, disable=disable):
            ode.reset()
            samples[i] = torchdiffeq.odeint(
                ode, x[i], t, rtol=rtol, atol=atol, method=solver,
                options={"min_step": 0.00001, "max_num_steps": max_steps,
                         "dtype": torch.float32 if torch.backends.mps.is_available() else torch.float64},
            )[-1]
    finally:
        ode.pbar.close()

    if callback is not None:
        callback({"x": samples, "i": n_steps - 1, "sigma": t_min,
                  "sigma_hat": t_min, "denoised": samples})
    return samples


# Select the original implementation where available; do not replace it globally.
def _get_native_sample_ode():
    try:
        from ldm_patched.k_diffusion import sampling
        if callable(getattr(sampling, "sample_ode", None)):
            return sampling.sample_ode
    except ImportError:
        pass
    return _ode_function_replay


class ReForgeODESampler:
    """Factory wrapper avoids binding to a WebUI-specific sampler at import time."""

    @staticmethod
    def make(sd_model):
        from modules.sd_samplers_kdiffusion import KDiffusionSampler

        class Sampler(KDiffusionSampler):
            def __init__(self, model):
                super().__init__(self._execute_ode, model, None)
                self.funcname = "reforge_ode_sampler"
                # AlterSampler.get_sigmas uses these members on reForge.
                self.sampler_name = "ode_custom"
                self._use_reforge_sigmas = False
                try:
                    from modules_forge.forge_alter_samplers import AlterSampler
                    self._use_reforge_sigmas = "get_sigmas" in vars(AlterSampler)
                except ImportError:
                    pass
                if self._use_reforge_sigmas:
                    self.unet = model.forge_objects.unet
                    self.model = model
                    self.scheduler_name = None

            def get_sigmas(self, p, steps):
                if self._use_reforge_sigmas:
                    from modules_forge.forge_alter_samplers import AlterSampler
                    return AlterSampler.get_sigmas(self, p, steps)
                return super().get_sigmas(p, steps)

            def _settings_for_pass(self):
                p = self.p
                default_solver = _valid_solver(_option(KEY_SOLVER, "dopri5"))
                txt_choice = getattr(p, "_rfo_txt_solver", USE_SAME)
                hr_choice = getattr(p, "_rfo_hr_solver", USE_SAME)

                if p.is_hr_pass:
                    # The Hires selection inherits the txt2img selection.
                    selected = txt_choice if hr_choice in (USE_SAME, LEGACY_SAME_TXT) else hr_choice
                    log_rtol = getattr(p, "_rfo_hr_rtol", _option(KEY_RTOL, -3.0))
                    log_atol = getattr(p, "_rfo_hr_atol", _option(KEY_ATOL, -4.0))
                else:
                    selected = txt_choice
                    log_rtol = getattr(p, "_rfo_txt_rtol", _option(KEY_RTOL, -3.0))
                    log_atol = getattr(p, "_rfo_txt_atol", _option(KEY_ATOL, -4.0))

                # No explicit UI override: use this extension's Settings,
                # never reForge's separate ode_custom_* options.
                if selected in (USE_SAME, LEGACY_SETTINGS, LEGACY_SAME_TXT):
                    method = default_solver
                    log_rtol = _option(KEY_RTOL, -3.0)
                    log_atol = _option(KEY_ATOL, -4.0)
                else:
                    method = _valid_solver(selected)
                max_steps = int(getattr(p, "_rfo_max_steps", _option(KEY_MAX, 250)))
                return method, float(log_rtol), float(log_atol), max_steps

            def _execute_ode(self, model, x, sigmas, extra_args=None, callback=None,
                             disable=None):
                method, log_rtol, log_atol, max_steps = self._settings_for_pass()
                prefix = "reForge-ODE hires" if self.p.is_hr_pass else "reForge-ODE"
                self.p.extra_generation_params[f"{prefix} solver"] = method
                if method in ADAPTIVE_SOLVERS:
                    self.p.extra_generation_params[f"{prefix} log_rtol"] = log_rtol
                    self.p.extra_generation_params[f"{prefix} log_atol"] = log_atol
                    self.p.extra_generation_params[f"{prefix} max_steps"] = max_steps

                # reForge passes None for fixed-grid rtol/atol; preserve that.
                rtol = 10 ** log_rtol if method in ADAPTIVE_SOLVERS else None
                atol = 10 ** log_atol if method in ADAPTIVE_SOLVERS else None
                return _get_native_sample_ode()(
                    model, x, sigmas, extra_args=extra_args, callback=callback,
                    disable=disable, solver=method, rtol=rtol, atol=atol,
                    max_steps=max_steps,
                )

            def sample(self, p, x, conditioning, unconditional_conditioning,
                       steps=None, image_conditioning=None):
                if self._use_reforge_sigmas:
                    self.scheduler_name = p.scheduler
                return super().sample(p, x, conditioning, unconditional_conditioning,
                                      steps=steps, image_conditioning=image_conditioning)

            def sample_img2img(self, p, x, noise, conditioning,
                               unconditional_conditioning, steps=None,
                               image_conditioning=None):
                if self._use_reforge_sigmas:
                    self.scheduler_name = p.scheduler
                return super().sample_img2img(
                    p, x, noise, conditioning, unconditional_conditioning,
                    steps=steps, image_conditioning=image_conditioning,
                )

        return Sampler(sd_model)


def _on_ui_settings():
    section = ("reforge-ode-sampler", "reForge-ODE Sampler")
    shared.opts.add_option(KEY_SOLVER, shared.OptionInfo(
        "dopri5", "Default ODE solver", gr.Dropdown,
        {"choices": SOLVERS}, section=section))
    shared.opts.add_option(KEY_RTOL, shared.OptionInfo(
        -3.0, "Default log relative tolerance", gr.Slider,
        {"minimum": -7.0, "maximum": 0.0, "step": 0.1}, section=section))
    shared.opts.add_option(KEY_ATOL, shared.OptionInfo(
        -4.0, "Default log absolute tolerance", gr.Slider,
        {"minimum": -7.0, "maximum": 0.0, "step": 0.1}, section=section))
    shared.opts.add_option(KEY_MAX, shared.OptionInfo(
        250, "Maximum ODE steps", gr.Slider,
        {"minimum": 1, "maximum": 500, "step": 1}, section=section))


script_callbacks.on_ui_settings(_on_ui_settings)


def _register():
    from modules import sd_samplers
    if not any(x.name == NAME for x in sd_samplers.all_samplers):
        sd_samplers.all_samplers.append(sd_samplers_common.SamplerData(
            NAME, ReForgeODESampler.make, ["reforge_ode_sampler"],
            {"scheduler": None},
        ))
    sd_samplers.all_samplers_map[NAME] = next(
        x for x in sd_samplers.all_samplers if x.name == NAME
    )
    sd_samplers.set_samplers()


def _on_model_loaded(model):
    try:
        _register()
    except Exception:
        LOG.exception("Failed to register on model load")


script_callbacks.on_model_loaded(_on_model_loaded)
try:
    _register()
except Exception:
    LOG.exception("Initial sampler registration failed")


try:
    from modules import scripts
    from modules.ui_components import InputAccordion

    class ReForgeODEScript(scripts.Script):
        def title(self):
            return NAME

        def show(self, is_img2img):
            return scripts.AlwaysVisible

        def ui(self, is_img2img):
            with InputAccordion(False, label=NAME) as enabled:
                with gr.Row():
                    txt_solver = gr.Dropdown(
                        choices=[USE_SAME] + SOLVERS, value=USE_SAME,
                        label="txt2img Solver",
                    )
                    hr_solver = gr.Dropdown(
                        choices=[USE_SAME] + SOLVERS, value=USE_SAME,
                        label="hires.fix Solver", visible=not is_img2img,
                    )
                with gr.Row():
                    txt_rtol = gr.Slider(
                        minimum=-7.0, maximum=0.0, step=0.1,
                        value=_option(KEY_RTOL, -3.0),
                        label="txt2img Log Relative Tolerance (10^x)",
                        interactive=False,
                    )
                    txt_atol = gr.Slider(
                        minimum=-7.0, maximum=0.0, step=0.1,
                        value=_option(KEY_ATOL, -4.0),
                        label="txt2img Log Absolute Tolerance (10^x)",
                        interactive=False,
                    )
                with gr.Row():
                    hr_rtol = gr.Slider(
                        minimum=-7.0, maximum=0.0, step=0.1,
                        value=_option(KEY_RTOL, -3.0),
                        label="hires.fix Log Relative Tolerance (10^x)",
                        interactive=False,
                        visible=not is_img2img,
                    )
                    hr_atol = gr.Slider(
                        minimum=-7.0, maximum=0.0, step=0.1,
                        value=_option(KEY_ATOL, -4.0),
                        label="hires.fix Log Absolute Tolerance (10^x)",
                        interactive=False,
                        visible=not is_img2img,
                    )
                with gr.Accordion("Advanced Settings", open=False):
                    max_steps = gr.Slider(
                        minimum=1, maximum=500, step=1,
                        value=_option(KEY_MAX, 250), label="Max ODE Steps",
                    )

                for slider in (txt_rtol, txt_atol, hr_rtol, hr_atol, max_steps):
                    slider.do_not_save_to_config = True

                def _tol_state(method):
                    return gr.update(interactive=method in ADAPTIVE_SOLVERS)

                txt_solver.change(_tol_state, inputs=[txt_solver], outputs=[txt_rtol])
                txt_solver.change(_tol_state, inputs=[txt_solver], outputs=[txt_atol])
                hr_solver.change(_tol_state, inputs=[hr_solver], outputs=[hr_rtol])
                hr_solver.change(_tol_state, inputs=[hr_solver], outputs=[hr_atol])

            self.infotext_fields = [
                (enabled, lambda d: "reForge-ODE solver" in d or "reForge-ODE hires solver" in d),
                (txt_solver, "reForge-ODE solver"),
                (hr_solver, "reForge-ODE hires solver"),
                (txt_rtol, "reForge-ODE log_rtol"),
                (txt_atol, "reForge-ODE log_atol"),
                (hr_rtol, "reForge-ODE hires log_rtol"),
                (hr_atol, "reForge-ODE hires log_atol"),
            ]
            return [enabled, txt_solver, hr_solver, txt_rtol, txt_atol,
                    hr_rtol, hr_atol, max_steps]

        def process(self, p, enabled, txt_solver, hr_solver, txt_rtol, txt_atol,
                    hr_rtol, hr_atol, max_steps):
            if not enabled:
                return
            p._rfo_txt_solver = txt_solver
            p._rfo_hr_solver = hr_solver
            p._rfo_txt_rtol = float(txt_rtol)
            p._rfo_txt_atol = float(txt_atol)
            p._rfo_hr_rtol = float(hr_rtol)
            p._rfo_hr_atol = float(hr_atol)
            p._rfo_max_steps = int(max_steps)
except ImportError:
    LOG.exception("Script UI components are unavailable")
