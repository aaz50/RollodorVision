"""
Tests for the Rollodor vision system.
These run WITHOUT a camera, model files, or any hardware.

Usage:
    python -m pytest test_detector.py -v
    # or just:
    python test_detector.py
"""

import numpy as np
import sys
import os

# Add parent to path so we can import detector
sys.path.insert(0, os.path.dirname(__file__))

from detector import (
    PersonDetector, SteerCommand, Detection, NavigationResult,
    CommandSender, CLASS_LABELS, PERSON_CLASS_ID,
)


# ---------------------------------------------------------------------------
# Test Detection dataclass logic
# ---------------------------------------------------------------------------

def test_detection_center_left():
    """A person on the left side of frame should have center_x < 0.35."""
    d = Detection(x=10, y=50, w=80, h=200, confidence=0.9,
                  center_x=0.1, area_ratio=0.05)
    assert d.center_x < 0.35, "Person at x=0.1 should be in LEFT zone"


def test_detection_center_right():
    """A person on the right side should have center_x > 0.65."""
    d = Detection(x=500, y=50, w=80, h=200, confidence=0.85,
                  center_x=0.85, area_ratio=0.05)
    assert d.center_x > 0.65, "Person at x=0.85 should be in RIGHT zone"


def test_detection_center_middle():
    """A person in the center should be between 0.35 and 0.65."""
    d = Detection(x=200, y=50, w=100, h=200, confidence=0.92,
                  center_x=0.5, area_ratio=0.08)
    assert 0.35 <= d.center_x <= 0.65, "Person at x=0.5 should be CENTER"


# ---------------------------------------------------------------------------
# Test navigation decision logic (no model needed)
# ---------------------------------------------------------------------------

class MockDetector(PersonDetector):
    """
    A detector that returns fake detections so we can test
    the navigation logic without loading a model or camera.
    """

    def __init__(self, fake_detections: list[Detection], **kwargs):
        super().__init__(**kwargs)
        self._fake = fake_detections

    def load_model(self):
        pass  # no-op

    def detect(self, frame: np.ndarray) -> list[Detection]:
        return self._fake


def make_frame(w=640, h=480) -> np.ndarray:
    """Create a blank test frame."""
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_no_person_returns_search():
    """When no person is detected, command should be SEARCH."""
    detector = MockDetector(fake_detections=[])
    result = detector.navigate(make_frame())
    assert result.command == SteerCommand.SEARCH
    assert result.target is None
    assert len(result.detections) == 0


def test_person_centered_returns_forward():
    """Person in center zone → FORWARD."""
    det = Detection(x=250, y=100, w=100, h=250, confidence=0.9,
                    center_x=0.5, area_ratio=0.08)
    detector = MockDetector(fake_detections=[det])
    result = detector.navigate(make_frame())
    assert result.command == SteerCommand.FORWARD
    assert result.target == det


def test_person_left_returns_left():
    """Person on left side → LEFT."""
    det = Detection(x=20, y=100, w=80, h=200, confidence=0.85,
                    center_x=0.15, area_ratio=0.05)
    detector = MockDetector(fake_detections=[det])
    result = detector.navigate(make_frame())
    assert result.command == SteerCommand.LEFT


def test_person_right_returns_right():
    """Person on right side → RIGHT."""
    det = Detection(x=500, y=100, w=80, h=200, confidence=0.88,
                    center_x=0.85, area_ratio=0.05)
    detector = MockDetector(fake_detections=[det])
    result = detector.navigate(make_frame())
    assert result.command == SteerCommand.RIGHT


def test_person_close_returns_stop():
    """Person filling >25% of frame → STOP (arrived)."""
    det = Detection(x=100, y=50, w=400, h=400, confidence=0.95,
                    center_x=0.5, area_ratio=0.52)
    detector = MockDetector(fake_detections=[det])
    result = detector.navigate(make_frame())
    assert result.command == SteerCommand.STOP


def test_multiple_people_targets_closest():
    """With multiple detections, the largest (closest) should be the target."""
    far_person = Detection(x=300, y=200, w=50, h=100, confidence=0.7,
                           center_x=0.5, area_ratio=0.02)
    close_person = Detection(x=200, y=100, w=200, h=350, confidence=0.9,
                             center_x=0.5, area_ratio=0.23)
    detector = MockDetector(fake_detections=[far_person, close_person])
    result = detector.navigate(make_frame())
    # Should target the close person (larger area ratio)
    assert result.target.area_ratio == close_person.area_ratio
    assert len(result.detections) == 2


def test_custom_center_zone():
    """Custom center zone should change steering thresholds."""
    # Narrow center zone: only 0.45-0.55 is "center"
    det = Detection(x=200, y=100, w=80, h=200, confidence=0.9,
                    center_x=0.4, area_ratio=0.05)
    detector = MockDetector(
        fake_detections=[det],
        center_zone=(0.45, 0.55),
    )
    result = detector.navigate(make_frame())
    # 0.4 is outside the narrow center zone, so should be LEFT
    assert result.command == SteerCommand.LEFT


def test_custom_close_distance():
    """Custom close distance threshold."""
    det = Detection(x=150, y=50, w=300, h=350, confidence=0.9,
                    center_x=0.5, area_ratio=0.35)
    # With default threshold of 0.25, this would STOP.
    # With a higher threshold, it should keep going FORWARD.
    detector = MockDetector(
        fake_detections=[det],
        close_distance_ratio=0.5,  # need 50% coverage to stop
    )
    result = detector.navigate(make_frame())
    assert result.command == SteerCommand.FORWARD


# ---------------------------------------------------------------------------
# Test CommandSender
# ---------------------------------------------------------------------------

def test_command_sender_deduplication():
    """Sender should not re-send the same command."""
    sender = CommandSender()
    sender.send(SteerCommand.FORWARD)
    assert sender.last_command == SteerCommand.FORWARD

    # Sending same command again should be a no-op (no crash)
    sender.send(SteerCommand.FORWARD)
    assert sender.last_command == SteerCommand.FORWARD

    # Different command updates
    sender.send(SteerCommand.LEFT)
    assert sender.last_command == SteerCommand.LEFT


# ---------------------------------------------------------------------------
# Test class labels
# ---------------------------------------------------------------------------

def test_person_class_id():
    """Verify person class index matches the label list."""
    assert CLASS_LABELS[PERSON_CLASS_ID] == "person"


def test_class_labels_count():
    """MobileNet-SSD has exactly 21 classes (including background)."""
    assert len(CLASS_LABELS) == 21


# ---------------------------------------------------------------------------
# Run tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_functions = [v for k, v in globals().items() if k.startswith("test_")]
    passed = 0
    failed = 0

    for fn in test_functions:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {fn.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR {fn.__name__}: {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    if failed > 0:
        sys.exit(1)