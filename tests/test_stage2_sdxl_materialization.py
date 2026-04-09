from pathlib import Path

from PIL import Image

from cspd_stage2.data import Stage2PairRecord
from cspd_stage2.families.sdxl.training import materialize_sdxl_training_dataset


def test_materialize_sdxl_training_dataset(tmp_path: Path) -> None:
    image_path = tmp_path / 'source.png'
    Image.new('RGB', (8, 8), color=(255, 0, 0)).save(image_path)

    pair = Stage2PairRecord(
        pair_id='p0',
        record_id='r0',
        sample_id='s0',
        relative_image_path='class_a/source.png',
        image_path=str(image_path),
        class_id=0,
        class_name_raw='class_a',
        class_name='class a',
        archetype='object',
        canonical_caption='a canonical red square',
    )

    summary = materialize_sdxl_training_dataset(pairs=[pair], output_dir=tmp_path / 'materialized')
    metadata_path = Path(summary['metadata_path'])
    rows = metadata_path.read_text(encoding='utf-8').strip().splitlines()

    assert summary['num_examples'] == 1
    assert metadata_path.exists()
    assert len(rows) == 1
    assert '"text": "a canonical red square"' in rows[0]
    copied_image = Path(summary['images_dir']) / Path(__import__('json').loads(rows[0])['file_name']).name
    assert copied_image.exists()
