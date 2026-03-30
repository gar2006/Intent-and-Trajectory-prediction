from __future__ import annotations

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for short trajectory sequences."""

    def __init__(self, d_model: int, max_len: int = 32) -> None:
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model)
        )

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class LSTMTrajectoryPredictor(nn.Module):
    """
    Simple encoder-decoder LSTM baseline for Phase 1.

    Input:
        history: [batch, past_steps, 6]
    Output:
        predicted future positions: [batch, future_steps, 2]
    """

    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 128,
        num_layers: int = 2,
        future_steps: int = 12,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.future_steps = future_steps
        self.hidden_dim = hidden_dim

        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.decoder = nn.LSTMCell(input_size=2, hidden_size=hidden_dim)
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        _, (hidden, cell) = self.encoder(history)

        decoder_hidden = hidden[-1]
        decoder_cell = cell[-1]

        decoder_input = history[:, -1, :2]
        predictions = []
        for _ in range(self.future_steps):
            decoder_hidden, decoder_cell = self.decoder(
                decoder_input, (decoder_hidden, decoder_cell)
            )
            next_xy = self.output_head(decoder_hidden)
            predictions.append(next_xy.unsqueeze(1))
            decoder_input = next_xy

        return torch.cat(predictions, dim=1)


class MapCNNEncoder(nn.Module):
    """Compact CNN encoder for 100x100 single-channel map patches."""

    def __init__(self, out_dim: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(64, out_dim)

    def forward(self, map_patch: torch.Tensor) -> torch.Tensor:
        features = self.features(map_patch).flatten(1)
        return self.proj(features)


class TransformerMapTrajectoryPredictor(nn.Module):
    """
    Phase 2 baseline:
    - Transformer encoder over observed pedestrian history
    - CNN encoder over local BEV map patch
    - Fused MLP decoder that predicts a single 12-step future trajectory

    We keep the output single-modal for now so training stays simple, while the
    encoder structure matches the later multimodal architecture.
    """

    def __init__(
        self,
        input_dim: int = 6,
        model_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 256,
        future_steps: int = 12,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.future_steps = future_steps

        self.history_proj = nn.Linear(input_dim, model_dim)
        self.pos_encoding = PositionalEncoding(model_dim, max_len=32)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
        )
        self.history_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.map_encoder = MapCNNEncoder(out_dim=model_dim)

        self.fusion = nn.Sequential(
            nn.Linear(model_dim * 2, model_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.decoder = nn.Sequential(
            nn.Linear(model_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, future_steps * 2),
        )

    def forward(self, history: torch.Tensor, map_patch: torch.Tensor) -> torch.Tensor:
        history_tokens = self.history_proj(history)
        history_tokens = self.pos_encoding(history_tokens)
        encoded_history = self.history_encoder(history_tokens)
        agent_embedding = encoded_history[:, -1]

        map_embedding = self.map_encoder(map_patch)
        fused = self.fusion(torch.cat([agent_embedding, map_embedding], dim=-1))
        pred = self.decoder(fused)
        return pred.view(history.size(0), self.future_steps, 2)


class SocialGATEncoder(nn.Module):
    """
    Lightweight GAT-style encoder over padded neighbor histories.

    Each neighbor trajectory is embedded independently, then attention weights are
    computed between the ego embedding and all valid neighbors.
    """

    def __init__(
        self,
        past_steps: int,
        input_dim: int,
        model_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.neighbor_encoder = nn.Sequential(
            nn.Linear(past_steps * input_dim, model_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, model_dim),
        )
        self.attn_mlp = nn.Sequential(
            nn.Linear(model_dim * 2, model_dim),
            nn.Tanh(),
            nn.Linear(model_dim, 1),
        )

    def forward(
        self,
        ego_embedding: torch.Tensor,
        neighbors: torch.Tensor,
        neighbor_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, max_neighbors, past_steps, input_dim = neighbors.shape
        flat_neighbors = neighbors.view(batch_size, max_neighbors, past_steps * input_dim)
        neighbor_embeddings = self.neighbor_encoder(flat_neighbors)

        ego_expanded = ego_embedding.unsqueeze(1).expand_as(neighbor_embeddings)
        attn_input = torch.cat([ego_expanded, neighbor_embeddings], dim=-1)
        scores = self.attn_mlp(attn_input).squeeze(-1)
        scores = scores.masked_fill(~neighbor_mask, float("-inf"))

        no_neighbor_rows = ~neighbor_mask.any(dim=1)
        if no_neighbor_rows.any():
            scores = scores.clone()
            scores[no_neighbor_rows] = 0.0

        attn = torch.softmax(scores, dim=-1)
        attn = attn * neighbor_mask.float()
        denom = attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        attn = attn / denom

        social_embedding = (attn.unsqueeze(-1) * neighbor_embeddings).sum(dim=1)
        return social_embedding


class SocialPoolingEncoder(nn.Module):
    """
    Simpler social encoder:
    - encode each neighbor independently
    - pool valid neighbors with mean or max pooling

    This is easier to explain than graph attention and works as a clean social
    pooling baseline for the project.
    """

    def __init__(
        self,
        past_steps: int,
        input_dim: int,
        model_dim: int,
        dropout: float = 0.1,
        pooling_type: str = "mean",
    ) -> None:
        super().__init__()
        if pooling_type not in {"mean", "max"}:
            raise ValueError("pooling_type must be 'mean' or 'max'")
        self.pooling_type = pooling_type
        self.neighbor_encoder = nn.Sequential(
            nn.Linear(past_steps * input_dim, model_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, model_dim),
        )

    def forward(
        self,
        ego_embedding: torch.Tensor,
        neighbors: torch.Tensor,
        neighbor_mask: torch.Tensor,
    ) -> torch.Tensor:
        del ego_embedding
        batch_size, max_neighbors, past_steps, input_dim = neighbors.shape
        flat_neighbors = neighbors.view(batch_size, max_neighbors, past_steps * input_dim)
        neighbor_embeddings = self.neighbor_encoder(flat_neighbors)

        if self.pooling_type == "max":
            masked = neighbor_embeddings.masked_fill(~neighbor_mask.unsqueeze(-1), float("-inf"))
            pooled = masked.max(dim=1).values
            no_neighbor_rows = ~neighbor_mask.any(dim=1)
            if no_neighbor_rows.any():
                pooled = pooled.clone()
                pooled[no_neighbor_rows] = 0.0
            return pooled

        masked = neighbor_embeddings * neighbor_mask.unsqueeze(-1).float()
        denom = neighbor_mask.sum(dim=1, keepdim=True).clamp_min(1).float()
        return masked.sum(dim=1) / denom


class TransformerMapSocialTrajectoryPredictor(nn.Module):
    """Phase 3 baseline with trajectory, map, and social fusion."""

    def __init__(
        self,
        input_dim: int = 6,
        model_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 256,
        past_steps: int = 4,
        future_steps: int = 12,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.future_steps = future_steps

        self.history_proj = nn.Linear(input_dim, model_dim)
        self.pos_encoding = PositionalEncoding(model_dim, max_len=32)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
        )
        self.history_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.map_encoder = MapCNNEncoder(out_dim=model_dim)
        self.social_encoder = SocialGATEncoder(
            past_steps=past_steps,
            input_dim=input_dim,
            model_dim=model_dim,
            dropout=dropout,
        )

        self.fusion = nn.Sequential(
            nn.Linear(model_dim * 3, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, model_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(model_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, future_steps * 2),
        )

    def forward(
        self,
        history: torch.Tensor,
        map_patch: torch.Tensor,
        neighbors: torch.Tensor,
        neighbor_mask: torch.Tensor,
    ) -> torch.Tensor:
        history_tokens = self.history_proj(history)
        history_tokens = self.pos_encoding(history_tokens)
        encoded_history = self.history_encoder(history_tokens)
        agent_embedding = encoded_history[:, -1]

        map_embedding = self.map_encoder(map_patch)
        social_embedding = self.social_encoder(agent_embedding, neighbors, neighbor_mask)
        fused = self.fusion(torch.cat([agent_embedding, map_embedding, social_embedding], dim=-1))
        pred = self.decoder(fused)
        return pred.view(history.size(0), self.future_steps, 2)


class TransformerMapSocialEncoder(nn.Module):
    """Shared encoder used by the later multimodal heads."""

    def __init__(
        self,
        input_dim: int = 6,
        model_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 256,
        past_steps: int = 4,
        dropout: float = 0.1,
        social_encoder_type: str = "gat",
        social_pooling_type: str = "mean",
    ) -> None:
        super().__init__()
        self.history_proj = nn.Linear(input_dim, model_dim)
        self.pos_encoding = PositionalEncoding(model_dim, max_len=32)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
        )
        self.history_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.map_encoder = MapCNNEncoder(out_dim=model_dim)
        if social_encoder_type == "gat":
            self.social_encoder = SocialGATEncoder(
                past_steps=past_steps,
                input_dim=input_dim,
                model_dim=model_dim,
                dropout=dropout,
            )
        elif social_encoder_type == "pool":
            self.social_encoder = SocialPoolingEncoder(
                past_steps=past_steps,
                input_dim=input_dim,
                model_dim=model_dim,
                dropout=dropout,
                pooling_type=social_pooling_type,
            )
        else:
            raise ValueError("social_encoder_type must be 'gat' or 'pool'")
        self.fusion = nn.Sequential(
            nn.Linear(model_dim * 3, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, model_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        history: torch.Tensor,
        map_patch: torch.Tensor,
        neighbors: torch.Tensor,
        neighbor_mask: torch.Tensor,
    ) -> torch.Tensor:
        history_tokens = self.history_proj(history)
        history_tokens = self.pos_encoding(history_tokens)
        encoded_history = self.history_encoder(history_tokens)
        agent_embedding = encoded_history[:, -1]

        map_embedding = self.map_encoder(map_patch)
        social_embedding = self.social_encoder(agent_embedding, neighbors, neighbor_mask)
        return self.fusion(torch.cat([agent_embedding, map_embedding, social_embedding], dim=-1))


class MultiModalTrajectoryDecoder(nn.Module):
    """Goal-conditioned decoder that emits one full trajectory per mode."""

    def __init__(self, model_dim: int, ff_dim: int, future_steps: int, num_modes: int, dropout: float) -> None:
        super().__init__()
        self.future_steps = future_steps
        self.num_modes = num_modes
        self.goal_proj = nn.Sequential(
            nn.Linear(2, model_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(model_dim * 2, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, future_steps * 2),
        )

    def forward(self, scene_embedding: torch.Tensor, endpoints: torch.Tensor) -> torch.Tensor:
        goal_embedding = self.goal_proj(endpoints)
        repeated_scene = scene_embedding.unsqueeze(1).expand(-1, self.num_modes, -1)
        fused = torch.cat([repeated_scene, goal_embedding], dim=-1)
        trajectories = self.decoder(fused)
        return trajectories.view(scene_embedding.size(0), self.num_modes, self.future_steps, 2)


class TransformerMapSocialMultiModalPredictor(nn.Module):
    """
    Phase 4 model:
    - shared trajectory/map/social encoder
    - intent classification head
    - K endpoint proposals
    - K trajectory probabilities
    - goal-conditioned decoder for K trajectories
    """

    def __init__(
        self,
        input_dim: int = 6,
        model_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 256,
        past_steps: int = 4,
        future_steps: int = 12,
        num_modes: int = 3,
        num_intents: int = 4,
        dropout: float = 0.1,
        social_encoder_type: str = "gat",
        social_pooling_type: str = "mean",
        use_agent_type_embedding: bool = False,
        agent_type_vocab_size: int = 2,
    ) -> None:
        super().__init__()
        self.num_modes = num_modes
        self.future_steps = future_steps
        self.use_agent_type_embedding = use_agent_type_embedding

        self.encoder = TransformerMapSocialEncoder(
            input_dim=input_dim,
            model_dim=model_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
            past_steps=past_steps,
            dropout=dropout,
            social_encoder_type=social_encoder_type,
            social_pooling_type=social_pooling_type,
        )
        self.agent_type_embedding: nn.Embedding | None = None
        if use_agent_type_embedding:
            self.agent_type_embedding = nn.Embedding(agent_type_vocab_size, model_dim)
        head_input_dim = model_dim * 2 if use_agent_type_embedding else model_dim
        self.endpoint_head = nn.Sequential(
            nn.Linear(head_input_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, num_modes * 2),
        )
        self.intent_head = nn.Sequential(
            nn.Linear(head_input_dim + 8, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, num_intents),
        )
        self.prob_head = nn.Sequential(
            nn.Linear(head_input_dim + 2, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, 1),
        )
        self.trajectory_decoder = MultiModalTrajectoryDecoder(
            model_dim=head_input_dim,
            ff_dim=ff_dim,
            future_steps=future_steps,
            num_modes=num_modes,
            dropout=dropout,
        )

    def forward(
        self,
        history: torch.Tensor,
        map_patch: torch.Tensor,
        neighbors: torch.Tensor,
        neighbor_mask: torch.Tensor,
        agent_type_id: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        scene_embedding = self.encoder(history, map_patch, neighbors, neighbor_mask)
        fused_embedding = scene_embedding
        if self.use_agent_type_embedding:
            if agent_type_id is None:
                raise ValueError("agent_type_id is required when use_agent_type_embedding=True")
            if self.agent_type_embedding is None:
                raise RuntimeError("agent_type_embedding is not initialized")
            type_embedding = self.agent_type_embedding(agent_type_id)
            fused_embedding = torch.cat([scene_embedding, type_embedding], dim=-1)

        endpoints = self.endpoint_head(fused_embedding).view(history.size(0), self.num_modes, 2)
        trajectories = self.trajectory_decoder(fused_embedding, endpoints)

        motion_summary = history[:, -1, 2:6]
        endpoint_mean = endpoints.mean(dim=1)
        endpoint_std = endpoints.std(dim=1, unbiased=False)
        intent_features = torch.cat(
            [fused_embedding, motion_summary, endpoint_mean, endpoint_std],
            dim=-1,
        )
        intent_logits = self.intent_head(intent_features)

        repeated_scene = fused_embedding.unsqueeze(1).expand(-1, self.num_modes, -1)
        prob_input = torch.cat([repeated_scene, endpoints], dim=-1)
        mode_logits = self.prob_head(prob_input).squeeze(-1)

        return {
            "intent_logits": intent_logits,
            "endpoints": endpoints,
            "mode_logits": mode_logits,
            "trajectories": trajectories,
        }
