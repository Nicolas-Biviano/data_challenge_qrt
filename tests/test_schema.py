from src.schema import Cols as SchemaCols
from src.utils import Cols as LegacyCols


def test_utils_compatibility_import_points_to_schema_module():
    assert LegacyCols is SchemaCols
    assert SchemaCols.__module__ == "src.schema"
