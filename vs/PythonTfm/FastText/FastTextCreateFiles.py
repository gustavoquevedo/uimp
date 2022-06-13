import fasttext
import spacy

from FastTextHelper import write_fasttext_records, write_fasttext_records_separate_oos
from CommonHelper import CommonHelper, ContentMode, ExecParams, SpacyTagsMode

exec_params = ExecParams.get_default_exec_params()

# Full
#============
if exec_params.process_full:
  write_fasttext_records_separate_oos('full', 'train')
  write_fasttext_records_separate_oos('full', 'test')

# Small
#============
if exec_params.process_small:
  write_fasttext_records_separate_oos('small', 'train')
  write_fasttext_records_separate_oos('small', 'test')

# Imbalanced
#============
if exec_params.process_imbalanced:
  write_fasttext_records_separate_oos('imbalanced', 'train')
  write_fasttext_records_separate_oos('imbalanced', 'test')

# OOS+
#============
if exec_params.process_oos_plus:
  write_fasttext_records_separate_oos('oos-plus', 'train')
  write_fasttext_records_separate_oos('oos-plus', 'test')

# Undersample
#============
if exec_params.process_undersample:
  write_fasttext_records('undersample', 'train')
  write_fasttext_records('undersample', 'test')

# Wiki-aug
#============
if exec_params.process_wiki_aug:
  write_fasttext_records('wiki-aug', 'train')
  write_fasttext_records('wiki-aug', 'test')