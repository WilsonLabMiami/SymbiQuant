# SymbiQuant (Previously Buchnearer)
Machine learning tool for polyploid independent population estimates for endosymbionts. The included dataset works to phenotype Buchnera and aphid bacteriocytes in DAPI stained confocal microscope images.
Please read ##Ref for information


To use Buchnera dataset, use this colab: https://colab.research.google.com/drive/1YZlQgm9Hf4qAFVPvb6JbNHlYyvS2cNN4#scrollTo=AcLEB5XJpH0b, 'GUI.py', and 'buchnera_metrics.py'.


To train for other endosymbiotic systems:

1. Annotate images of endosymbiotic tissue using Labelme (https://github.com/wkentaro/labelme)

2. Divide images and annotations into tiles using 'split_json.py'

3. Randomly divide tiles into training, test, and validation datasets, and convert j

4. Train maskRCNN on your datasets using Detectron2. A basic Detectron2 tutorial is here: https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5
The specific training settings we used are in 'trainer_settings.py', although these settings may need adjustments for other endosymbiotic systems i.e. non-spherical endosymbionts. Our data augmentation settings are in 'trainer_augmentation.py'.

5. The resultant model weights (the '.pth' file) can be used in place of ours in a detector - see 'detector_example.py' and 'Buchnearer.py' for examples.

6. Assess performance of your model - i.e. prediction & recall - and use a GUI (like 'GUI.py') to curate results. GUI may need to be altered for non-spherical endosymbionts.

