"""Chay lai RIENG mot so thuat toan cua o 6c NGOAI notebook, gop vao file ket qua (kq_6c_*.pkl).

Dung khi thoi gian do trong notebook bi nhieu (may ban viec khac luc chay). Script lay DUNG ma tu
bai_lam.ipynb (cac ham thuat toan, hang so luoi, o 6c) nen ket qua giong het chay trong notebook,
chi khong chay cac o quet / ve hinh. Sau do mo notebook voi CHAY_LAI_6C = False -> o 6c doc file da gop.

    python chay_6c_rieng.py "GD cố định" "GD backtracking" "Nesterov"
    python chay_6c_rieng.py "ISTA|t=6/L"        # chi MOT setup (dang "Thuat toan|setup")

Chay TUAN TU, khong chay song song nhieu ban (tranh CPU -> sai thoi gian).
"""
import json, re, sys, time, warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")      # console Windows mac dinh cp1252 -> khong in duoc tieng Viet

THUAT_TOAN = sys.argv[1:]
assert THUAT_TOAN, "can it nhat mot ten thuat toan (nhu cot 'Thuat toan' o bang 6c)"
nb = json.load(open("bai_lam.ipynb", encoding="utf-8"))
O = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def o_co(dau):
    """O code DAU TIEN chua chuoi `dau`."""
    return next(s for s in O if dau in s)


g = {}
for dau in ["import pandas as pd", "def lam_sach", "def sigmoid", "CAU HINH CHAY", "def gradient_descent(",
            "def gradient_descent_bt"]:
    exec(o_co(dau), g)
# hang so luoi nam trong cac o quet -> chi lay dung dong gan, khong chay ca o quet
for ten in ["BUOC", "BUOC_THEM", "BUOC_NES_LON", "BUOC_NES", "BUOC_PX_LON", "BUOC_ISTA", "BUOC_FISTA", "BUOC_GD_LON", "BUOC_GD", "BETA", "C_ARMIJO", "LUOI_BT", "BUOC_NT", "BETA_NT", "ETA", "ETA_SG"]:
    dong = next(l for s in O for l in s.splitlines() if re.match(r"%s\s*=" % ten, l))
    exec(dong, g)
for dau in ["def gd_nesterov", "def newton", "def sgd(", "def subgradient", "def prox_l1"]:
    exec(o_co(dau), g)

o6c = o_co("6c. KET QUA CHINH")
o6c, so = re.subn(r"^CHAY_LAI_6C = .*$", "CHAY_LAI_6C = %r" % THUAT_TOAN, o6c, count=1, flags=re.M)
assert so == 1 and "CHAY_DAY_DU = True" in o6c, "o 6c phai dang o che do CHAY_DAY_DU = True"
t0 = time.time()
print("Bat dau %s, chay lai: %s" % (time.strftime("%Y-%m-%d %H:%M"), ", ".join(THUAT_TOAN)), flush=True)
exec(o6c, g)
print("Xong %s (%.0f phut)" % (time.strftime("%Y-%m-%d %H:%M"), (time.time() - t0) / 60))
