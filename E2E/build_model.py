import torch
from backbone.backbone import  load_backbone
from detector.detector import load_detector


class TAL_model(torch.nn.Module):

    def __init__(self):
        self.chunk_size = chunk_size
        self.sampling_ratio = sampling_ratio
          
        self.feature_extractor = load_backbone()
        self.base_detector = load_detector()

    def forward(self, video_data, feat_grad=None, stage=0):
        
        if stage == 1:  # sequentially forward the backbone
            video_feat = self.forward_stage_1(video_data)
            return video_feat

        elif stage == 2:  # forward and backward the detector
            video_feat = video_data
            det_pred = self.base_detector(video_feat)
            return det_pred

        elif stage == 3:  # sequentially backward the backbone with sampled data
                self.forward_stage_3(video_data, feat_grad=feat_grad)

        elif stage == 0:  # this is for inference
                video_feat = self.forward_stage_1(video_data)
                det_pred = self.base_detector(video_feat)
                return det_pred


    def forward_stage_1(self, frames):
        # sequentially forward backbone
        chunk_num = frames.shape[1] // self.chunk_size  # frames [B,N,C,T,H,W]
        video_feat = []
        with torch.set_grad_enabled(False):
            for mini_frames in torch.chunk(frames, chunk_num, dim=1):
                video_feat.append(self.feature_extractor(mini_frames))
        video_feat = torch.cat(video_feat, dim=2)

        # clean cache
        video_feat = video_feat.detach()
        torch.cuda.empty_cache()
        return video_feat

    def forward_stage_3(self, video_data, feat_grad):

        B, T, C, L, H, W = video_data.shape  # batch, snippet length, 3, clip length, h, w

        # sample the snippets

        chunk_num = int(T * self.sampling_ratio / self.chunk_size + 0.99)
        assert chunk_num > 0 and chunk_num * self.chunk_size <= T

        # random sampling

        noise = torch.rand(B, T, device=video_data.device)  # noise in [0, 1]

        # sort noise for each sample

        ids_shuffle = torch.argsort(noise, dim=1)

        for chunk_idx in range(chunk_num):
            snippet_idx = ids_shuffle[:, chunk_idx * self.chunk_size : (chunk_idx + 1) * self.chunk_size]

            video_data_chunk = torch.gather(
                video_data,
                dim=1,
                index=snippet_idx.view(B, self.chunk_size, 1, 1, 1, 1).repeat(1, 1, C, L, H, W),
            )
            feat_grad_chunk = torch.gather(
                feat_grad,
                dim=2,
                index=snippet_idx.view(B, 1, self.chunk_size).repeat(1, feat_grad.shape[1], 1),
            )
            self.feature_extractor = self.feature_extractor.train()
            with torch.set_grad_enabled(train):
                video_feat_chunk = self.feature_extractor(video_data_chunk)
            assert video_feat_chunk.shape == feat_grad_chunk.shape

            # accumulate grad
            video_feat_chunk.backward(gradient=feat_grad_chunk)
