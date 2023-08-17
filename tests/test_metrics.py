import torch
import tensorflow as tf
from einops import rearrange
from fusionlab.metrics import (
    DiceScore, 
    IoUScore,
)
from pytest import approx

EPS = 1e-6

class Data:
    def __init__(self):
        self.pred = [[
            [1., 2., 3., 4.],
            [2., 6., 4., 4.],
            [9., 6., 3., 4.]
        ]]
        self.target = [[2, 1, 0, 2]]


class BinaryData:
    def __init__(self):
        self.pred = [0.4, 0.2, 0.3, 0.5]
        self.target = [0, 1, 0, 1]


class TestMetrics:
    def test_dice_score(self):
        # Multi class
        data = Data()
        true_scores = [0.27264893, 0.41188607, 0.65953231]
        true_score_mean = 0.44802245
        # PT
        pred = torch.tensor(data.pred).view(1, 3, 4)
        true = torch.tensor(data.target).view(1, 4)
        scores = DiceScore("multiclass")(pred, true)
        assert scores.mean() == approx(true_score_mean, EPS)
        assert scores == approx(true_scores, EPS)

        # TODO: Add binary class test
        data = BinaryData()
        true_loss = 0.46044695377349854
    
    def test_iou_score(self):
        # multi class
        true_loss = 0.6969285607337952
        true_scores = [0.15784223, 0.25935552, 0.49201655]
        true_score_mean = 0.30307144
        data = Data()

        # PT
        pred = torch.tensor(data.pred).view(1, 3, 4)
        true = torch.tensor(data.target).view(1, 4)
        scores = IoUScore("multiclass", from_logits=True)(pred, true)
        assert scores.mean() == approx(true_score_mean, EPS)
        assert scores == approx(true_scores, EPS)

        # TODO: Add binary class test

