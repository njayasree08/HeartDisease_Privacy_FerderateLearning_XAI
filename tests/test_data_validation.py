import io, pandas as pd, pytest
from src import config
from src.data import loader
from src.data.validation import validate_patient_input, validate_schema

GOOD = dict(age=55, sex=1, cp=0, trestbps=130, chol=240, fbs=0, restecg=1, thalach=150, exang=0, oldpeak=1.0, slope=2, ca=0, thal=2)


def test_schema_ok(frame):
    assert validate_schema(frame).ok


def test_missing_column_and_non_numeric(frame):
    assert not validate_schema(frame.drop(columns=["chol"])).ok
    bad = frame.copy(); bad["age"] = bad["age"].astype(str) + "x"
    assert not validate_schema(bad).ok


def test_target_conversion():
    df = pd.DataFrame({"num": [0, 1, 2, 4, 0], "cp": [1, 2, 3, 4, 1]})
    assert loader.to_binary_target(df, "num")["target"].tolist() == [0, 1, 1, 1, 0]
    assert loader.to_binary_target(pd.DataFrame({"target": [0, 1, 1]}), "target", invert=True)["target"].tolist() == [1, 0, 0]


def test_target_never_in_features():
    assert "target" not in config.FEATURES and "num" not in config.FEATURES


def test_missing_dataset_raises(tmp_path):
    with pytest.raises(loader.DatasetError):
        loader.load_csv(tmp_path / "nope.csv")


def test_upload_like_object_and_placeholder_codes():
    row = ",".join(["age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,target"])
    csv = row + "\n50,1,0,120,200,0,1,150,0,1.0,2,4,0,1\n60,0,1,130,210,0,1,140,0,0.5,1,0,2,0\n"
    df, info = loader.load_prepared(io.StringIO(csv))
    assert pd.isna(df.loc[0, "ca"]) and pd.isna(df.loc[0, "thal"])  # placeholder codes -> real NaN


def test_patient_input_validation():
    assert validate_patient_input(GOOD) == []
    assert validate_patient_input({**GOOD, "age": 500})
    assert validate_patient_input({**GOOD, "cp": 9})
    assert validate_patient_input({**GOOD, "chol": None})
    assert validate_patient_input({k: v for k, v in GOOD.items() if k != "sex"})


def test_excel_loading_and_support(tmp_path):
    df_orig = pd.DataFrame({
        "age": [55, 60], "sex": [1, 0], "cp": [0, 1], "trestbps": [130, 140],
        "chol": [240, 260], "fbs": [0, 1], "restecg": [1, 0], "thalach": [150, 130],
        "exang": [0, 1], "oldpeak": [1.0, 2.5], "slope": [2, 1], "ca": [0, 2],
        "thal": [2, 3], "target": [1, 0]
    })
    xlsx_file = tmp_path / "test_data.xlsx"
    df_orig.to_excel(xlsx_file, index=False)
    loaded = loader.load_dataset_file(xlsx_file)
    assert len(loaded) == 2 and "target" in loaded.columns
    assert loaded["age"].tolist() == [55, 60]


def test_verify_target_semantics():
    # Synthetic dataset where target 1 has high oldpeak and high exang (disease = 1)
    df_std = pd.DataFrame({
        "oldpeak": [0.0, 0.2, 2.5, 3.0],
        "exang": [0, 0, 1, 1],
        "ca": [0, 0, 2, 3],
        "thalach": [180, 175, 120, 110],
        "age": [35, 40, 65, 70],
        "target": [0, 0, 1, 1]
    })
    verif = loader.verify_target_semantics(df_std, "target")
    assert verif["verified"] is True
    assert verif["recommended_invert"] is False  # Already positive for disease
    assert "oldpeak" in verif["correlations"]

    # Inverted dataset where target 0 has high oldpeak and high exang
    df_inv = pd.DataFrame({
        "oldpeak": [3.0, 2.5, 0.2, 0.0],
        "exang": [1, 1, 0, 0],
        "ca": [3, 2, 0, 0],
        "thalach": [110, 120, 175, 180],
        "age": [70, 65, 40, 35],
        "target": [0, 0, 1, 1]
    })
    verif_inv = loader.verify_target_semantics(df_inv, "target")
    assert verif_inv["verified"] is True
    assert verif_inv["recommended_invert"] is True  # Needs inversion so 1=disease

