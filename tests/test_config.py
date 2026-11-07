from cattle_breed import config
from cattle_breed.config import CLASS_TO_IDX, CLASSES, DATASET_CLASSES, to_canonical


def test_classes_fixed_order():
    assert CLASSES == ("Holstein cross", "Local", "Red Chittagong", "Sahiwal", "Sindhi")
    assert [CLASS_TO_IDX[c] for c in CLASSES] == [0, 1, 2, 3, 4]


def test_cid_raw_labels_map():
    assert to_canonical("HOSTINE_CROSS") == "Holstein cross"
    assert to_canonical("LOCAL") == "Local"
    assert to_canonical("RED_CHITTAGONG") == "Red Chittagong"
    assert to_canonical("SAHIWAL") == "Sahiwal"
    assert to_canonical("SINDHI") == "Sindhi"
    for dropped in ("PABNA_BREED", "BRAHMA", "MIR_KADIM"):
        assert to_canonical(dropped) is None


def test_external_labels_map():
    assert to_canonical("Holstain Friesian", "mendeley") == "Holstein cross"  # real folder name (misspelt)
    assert to_canonical("Local Deshi", "mendeley") == "Local"
    assert to_canonical("Jersey", "mendeley") is None
    assert to_canonical("Red_Sindhi", "kaggle") == "Sindhi"
    assert to_canonical("Holstein_Friesian", "kaggle") == "Holstein cross"
    assert to_canonical("Gir", "kaggle") is None


def test_dataset_restriction():
    assert to_canonical("local", "kaggle") is None  # a valid class, but not one Kaggle contains
    for classes in DATASET_CLASSES.values():
        assert set(classes) <= set(CLASSES)


def test_training_recipe_matches_proposal():
    assert (config.HEAD_LR, config.HEAD_EPOCHS, config.FT_LR, config.FT_EPOCHS) == (1e-3, 10, 1e-5, 20)
    assert (config.BATCH_SIZE, config.DROPOUT, config.LABEL_SMOOTHING) == (32, 0.3, 0.1)
    assert set(config.CORRUPTIONS) == {"blur", "dark", "occlude"}
