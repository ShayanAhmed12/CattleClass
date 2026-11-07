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
    assert to_canonical("Holstein Friesian", "mendeley") == "Holstein cross"
    assert to_canonical("Local Deshi", "mendeley") == "Local"
    assert to_canonical("Jersey", "mendeley") is None
    assert to_canonical("Red_Sindhi", "kaggle") == "Sindhi"
    assert to_canonical("Holstein_Friesian", "kaggle") == "Holstein cross"
    assert to_canonical("Gir", "kaggle") is None


def test_dataset_restriction():
    # 'Local' is a valid class but the Kaggle set must not produce it.
    assert to_canonical("local", "kaggle") is None
    for classes in DATASET_CLASSES.values():
        assert set(classes) <= set(CLASSES)


def test_train_config_defaults_match_plan():
    cfg = config.TrainConfig()
    assert (cfg.head_lr, cfg.head_epochs, cfg.head_patience) == (1e-3, 10, 3)
    assert (cfg.ft_lr, cfg.ft_epochs, cfg.ft_patience) == (1e-5, 20, 4)
    assert (cfg.batch_size, cfg.dropout, cfg.label_smoothing) == (32, 0.3, 0.1)


def test_corruption_table():
    assert set(config.CORRUPTIONS) == {"blur", "dark", "occlude"}
    assert all(set(v) == {1, 2} for v in config.CORRUPTIONS.values())
