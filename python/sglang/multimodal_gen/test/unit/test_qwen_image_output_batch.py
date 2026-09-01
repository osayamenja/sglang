import unittest
from types import SimpleNamespace

import torch

from sglang.multimodal_gen.configs.pipeline_configs.qwen_image import (
    QwenImagePipelineConfig,
)


class TestQwenImageOutputBatch(unittest.TestCase):
    def test_repeats_conditioning_for_multiple_outputs(self):
        config = QwenImagePipelineConfig()
        prompt_embeds = torch.arange(2 * 3 * 4).reshape(2, 3, 4)
        prompt_mask = torch.tensor(
            [[True, True, False], [True, True, True]], dtype=torch.bool
        )
        batch = SimpleNamespace(
            height=1024,
            width=1024,
            num_outputs_per_prompt=2,
            raw_latent_shape=(4, 1, 64, 64),
            prompt_embeds=[prompt_embeds],
            negative_prompt_embeds=[prompt_embeds + 100],
            prompt_embeds_mask=[prompt_mask],
            negative_prompt_embeds_mask=[prompt_mask],
            prompt_seq_lens=[[2, 3]],
            negative_prompt_seq_lens=[[2, 3]],
        )

        expanded = config.get_pos_prompt_embeds(batch)[0]
        torch.testing.assert_close(expanded, prompt_embeds.repeat_interleave(2, dim=0))

        kwargs = config.prepare_pos_cond_kwargs(
            batch, torch.device("cpu"), rotary_emb=None, dtype=torch.bfloat16
        )
        self.assertEqual(kwargs["txt_seq_lens"], [2, 2, 3, 3])
        torch.testing.assert_close(
            kwargs["encoder_hidden_states_mask"],
            prompt_mask.repeat_interleave(2, dim=0),
        )
        self.assertEqual(len(kwargs["img_shapes"]), 4)


if __name__ == "__main__":
    unittest.main()
