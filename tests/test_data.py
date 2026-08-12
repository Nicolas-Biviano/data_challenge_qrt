from src.data import ChallengeDataLoader as NewLoader
from src.dataloader import ChallengeDataLoader as LegacyLoader


def test_dataloader_compatibility_import_points_to_data_module():
    assert LegacyLoader is NewLoader
    assert NewLoader.__module__ == "src.data"
