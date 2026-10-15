# Attribution and source notices

This extension includes a modified ODE fallback derived from
[Panchovix/stable-diffusion-webui-reForge](https://github.com/Panchovix/stable-diffusion-webui-reForge),
`ldm_patched/k_diffusion/sampling.py`, with reference revision
[`739b2e1d9ab63160eaff9c8f73172c8da68424e1`](https://github.com/Panchovix/stable-diffusion-webui-reForge/blob/739b2e1d9ab63160eaff9c8f73172c8da68424e1/ldm_patched/k_diffusion/sampling.py).
reForge is distributed under the GNU Affero General Public License, version 3;
its license text, including its existing notice, is retained in [LICENSE](LICENSE).

The ODE core originates from redhottensors' [ComfyUI-ODE](https://github.com/redhottensors/ComfyUI-ODE),
`nodes/nodes_ode.py`, distributed under the GNU General Public License, version 3.
Its license text is retained in [licenses/ComfyUI-ODE-GPL-3.0.txt](licenses/ComfyUI-ODE-GPL-3.0.txt).
Upstream attribution is retained; no ownership of upstream work is claimed.

The extension's additions include independent sampler registration and settings,
per-pass solver/tolerance controls, PNG infotext handling, native ODE/sigma-path
selection, and packaging. The fallback reorganizes the upstream ODE classes into
one function and adds progress-bar cleanup. On reForge, the native implementation
is preferred when available.

This extension is distributed under GNU AGPL v3; see [LICENSE](LICENSE).
The included GPL v3 upstream material remains subject to its original license.
The complete corresponding source for this extension is provided in this repository.

Thanks to redhottensors for the original ComfyUI-ODE implementation and to
Panchovix for porting and integrating ODE sampling into reForge.
