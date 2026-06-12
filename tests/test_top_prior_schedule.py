import pytest

from eps_seg.modules.top_prior import TopPriorSchedulingMixin, build_top_prior_params


class DummyTopPriorLayer(TopPriorSchedulingMixin):
    def __init__(
        self,
        *,
        mu_init: float,
        supervised_target: float,
        semisupervised_target: float,
        step_epochs: int = 2,
        mode: str = "supervised",
    ):
        self.is_top_layer = True
        self.training_mode = mode
        prior_params, mu_mask = build_top_prior_params(
            n_components=2,
            top_prior_param_shape=(1, 8, 1, 1),
            conv_mult=2,
            learn_top_prior=False,
            mu_init=mu_init,
        )
        self.top_prior_params = prior_params
        self.top_prior_mu_mask = mu_mask
        self._init_top_prior_schedule(
            schedule="dynamic",
            mu_init=mu_init,
            mu_supervised_max=supervised_target,
            mu_semisupervised_min=semisupervised_target,
            mu_step_epochs=step_epochs,
        )

    def update_mode(self, mode: str) -> None:
        self.training_mode = mode
        self._schedule_phase_start_epoch = None
        self._schedule_phase_start_mu = None
        self._schedule_phase_mode = None


def test_dynamic_top_prior_supervised_can_increase_toward_target():
    layer = DummyTopPriorLayer(mu_init=1.0, supervised_target=4.0, semisupervised_target=4.0)

    layer.update_top_prior_scheduler(0)
    assert layer.get_top_prior_mu_value() == pytest.approx(1.0)

    layer.update_top_prior_scheduler(2)
    assert layer.get_top_prior_mu_value() == pytest.approx(2.0)

    layer.update_top_prior_scheduler(20)
    assert layer.get_top_prior_mu_value() == pytest.approx(4.0)


def test_dynamic_top_prior_supervised_can_decrease_toward_target():
    layer = DummyTopPriorLayer(mu_init=10.0, supervised_target=5.0, semisupervised_target=5.0)

    layer.update_top_prior_scheduler(0)
    assert layer.get_top_prior_mu_value() == pytest.approx(10.0)

    layer.update_top_prior_scheduler(2)
    assert layer.get_top_prior_mu_value() == pytest.approx(9.0)

    layer.update_top_prior_scheduler(20)
    assert layer.get_top_prior_mu_value() == pytest.approx(5.0)


def test_dynamic_top_prior_semisupervised_can_decrease_toward_target():
    layer = DummyTopPriorLayer(mu_init=1.0, supervised_target=4.0, semisupervised_target=2.0)
    layer.update_top_prior_scheduler(0)
    layer.update_top_prior_scheduler(20)
    assert layer.get_top_prior_mu_value() == pytest.approx(4.0)

    layer.update_mode("semisupervised")
    layer.update_top_prior_scheduler(20)
    assert layer.get_top_prior_mu_value() == pytest.approx(4.0)

    layer.update_top_prior_scheduler(22)
    assert layer.get_top_prior_mu_value() == pytest.approx(3.0)

    layer.update_top_prior_scheduler(40)
    assert layer.get_top_prior_mu_value() == pytest.approx(2.0)


def test_dynamic_top_prior_semisupervised_keeps_carried_mu_when_target_is_higher():
    layer = DummyTopPriorLayer(mu_init=1.0, supervised_target=10.0, semisupervised_target=10.0)
    layer.update_top_prior_scheduler(0)
    layer.update_top_prior_scheduler(4)
    assert layer.get_top_prior_mu_value() == pytest.approx(3.0)

    layer.update_mode("semisupervised")
    layer.update_top_prior_scheduler(4)
    assert layer.get_top_prior_mu_value() == pytest.approx(3.0)

    layer.update_top_prior_scheduler(20)
    assert layer.get_top_prior_mu_value() == pytest.approx(3.0)
