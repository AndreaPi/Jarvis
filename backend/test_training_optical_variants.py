import copy
import json
import tempfile
import unittest
from pathlib import Path
from PIL import Image
from backend import train_full_image_digit_detector as training


class OpticalVariantTests(unittest.TestCase):
  def test_review_lineage_and_geometry_are_required_before_replacing_train_crop(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      dataset, review = root / 'dataset', root / 'pilot'
      for base in (dataset / 'images/train', dataset / 'labels/train', review / 'review/images', review / 'review/labels'):
        base.mkdir(parents=True)
      source = dataset / 'images/train/source.JPEG'
      name = 'source__digit_p2_d6_r00.JPEG'
      crop = source.with_name(name)
      variant = review / 'review/images' / name
      for path, color in ((source, 'gray'), (crop, 'gray'), (variant, 'silver')):
        Image.new('RGB', (40, 60), color).save(path)
      label = dataset / 'labels/train' / (Path(name).stem + '.txt')
      label.write_text('6 0.5 0.5 0.4 0.5\n')
      (review / 'review/labels' / label.name).write_bytes(label.read_bytes())
      record = dict(filename=name, source_filename=source.name, class_id=6, source_fold=1,
                    qa_status='passed', source_sha256=training.file_sha256(source),
                    original_crop_sha256=training.file_sha256(crop), synthetic_sha256=training.file_sha256(variant),
                    label_sha256=training.file_sha256(label), image_size=[40, 60])
      qa = review / 'qa.json'
      qa.write_text(json.dumps(dict(status='passed', geometry_changed=False, label_changes=0, checked_images={name: training.file_sha256(variant)})))
      manifest = dict(status='agent_qa_passed', fold=0, count=1, method='optical', records=[record],
                      qa=dict(report_sha256=training.file_sha256(qa)))
      manifest_path = review / 'manifest.json'
      before = training.materialized_dataset_sha256(dataset)
      cases = [('pending', 'status', 'pending'), ('wrong_fold', 'fold', 2),
               ('count', 'count', 2), ('held_out', 'source_fold', 0),
               ('changed_image', 'synthetic_sha256', 'changed'),
               ('dimensions', 'image_size', [41, 60]), ('unreviewed_record', 'qa_status', 'pending'),
               ('path', 'filename', '../escape.JPEG')]
      for reason, key, value in cases:
        with self.subTest(reason=reason):
          bad = copy.deepcopy(manifest)
          (bad if key in ('status', 'fold', 'count') else bad['records'][0])[key] = value
          manifest_path.write_text(json.dumps(bad))
          with self.assertRaises(ValueError):
            training.apply_training_optical_variants(dataset, manifest_path, {'source.JPEG': 1}, 0)
          self.assertEqual(training.materialized_dataset_sha256(dataset), before)
      manifest_path.write_text(json.dumps(manifest))
      for folds in ({'source.JPEG': 0}, {}):
        with self.assertRaises(ValueError):
          training.apply_training_optical_variants(dataset, manifest_path, folds, 0)
      result = training.apply_training_optical_variants(dataset, manifest_path, {'source.JPEG': 1}, 0)
      self.assertEqual(result['replaced_images'], 1)
      self.assertEqual(crop.read_bytes(), variant.read_bytes())
      self.assertEqual(training.file_sha256(source), record['source_sha256'])
      self.assertEqual(training.file_sha256(label), record['label_sha256'])

  def test_resume_rejects_removed_or_changed_optical_recipe(self):
    from backend.test_train_full_image_digit_detector import resume_provenance
    with tempfile.TemporaryDirectory() as tmp:
      path = Path(tmp)
      original = resume_provenance()
      original['train_optical_variants'] = {'manifest_sha256': 'frozen'}
      (path / 'dataset_provenance.json').write_text(json.dumps(original))
      training.validate_resume_provenance(path, original)
      for replacement in (None, {'manifest_sha256': 'changed'}):
        changed = copy.deepcopy(original)
        if replacement is None:
          del changed['train_optical_variants']
        else:
          changed['train_optical_variants'] = replacement
        with self.assertRaisesRegex(ValueError, 'train_optical_variants'):
          training.validate_resume_provenance(path, changed)
