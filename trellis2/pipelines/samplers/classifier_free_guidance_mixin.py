from typing import *


class ClassifierFreeGuidanceSamplerMixin:
    """
    A mixin class for samplers that apply classifier-free guidance.
    """

    def _inference_model(self, model, x_t, t, cond, step_idx, neg_cond, guidance_strength, guidance_rescale=0.0, **kwargs):
        symmetric_tar_cond = kwargs.pop("cfg_symmetric_tar_cond", False)
        legacy_cfg_formula = kwargs.pop("legacy_cfg_formula", False)

        def _neg_kwargs():
            neg_kwargs = dict(kwargs)
            if symmetric_tar_cond and "tar_cond" in neg_kwargs:
                neg_kwargs["tar_cond"] = neg_cond
            return neg_kwargs

        if guidance_strength == 1:
            return super()._inference_model(model, x_t, t, cond, step_idx, **kwargs)
        elif guidance_strength == 0:
            return super()._inference_model(model, x_t, t, neg_cond, step_idx, **_neg_kwargs())
        else:
            pred_pos = super()._inference_model(model, x_t, t, cond, step_idx, **kwargs)
            pred_neg = super()._inference_model(model, x_t, t, neg_cond, step_idx, **_neg_kwargs())
            if legacy_cfg_formula:
                pred = (1 + guidance_strength) * pred_pos - guidance_strength * pred_neg
            else:
                pred = guidance_strength * pred_pos + (1 - guidance_strength) * pred_neg
            
            # CFG rescale
            if guidance_rescale > 0:
                x_0_pos = self._pred_to_xstart(x_t, t, pred_pos)
                x_0_cfg = self._pred_to_xstart(x_t, t, pred)
                std_pos = x_0_pos.std(dim=list(range(1, x_0_pos.ndim)), keepdim=True)
                std_cfg = x_0_cfg.std(dim=list(range(1, x_0_cfg.ndim)), keepdim=True)
                x_0_rescaled = x_0_cfg * (std_pos / std_cfg)
                x_0 = guidance_rescale * x_0_rescaled + (1 - guidance_rescale) * x_0_cfg
                pred = self._xstart_to_pred(x_t, t, x_0)
                
            return pred
