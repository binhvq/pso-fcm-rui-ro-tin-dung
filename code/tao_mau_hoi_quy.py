# -*- coding: utf-8 -*-
"""Sinh tệp mẫu cho phép thử hồi quy tiền xử lý (kiem_thu_tien_xu_ly.test_hoi_quy_X_khong_doi).

    python3 tao_mau_hoi_quy.py

Ghi mau_kiem_thu/hoi-quy-X.npz: ma trận X của mọi bộ trong tien_xu_ly.DANH_SACH theo cách tiền xử lý hiện tại, riêng
Australian lấy theo cách cũ (có log có dấu) để phép thử khẳng định việc bỏ log ở bộ này là thay đổi có chủ đích. Tệp mẫu
ban đầu nằm ở thư mục tạm và đã mất; bản này tạo lại ngày 20/10/2026, sau khi đối chiếu X của mọi bộ trùng từng bit với
tien_xu_ly.py ở bản sao lưu ngày 15/10. Chỉ chạy lại khi cố ý đổi tiền xử
lý, vì chạy lại sẽ làm phép thử hồi quy luôn đạt.
"""

from __future__ import annotations

import os

import numpy as np

import tien_xu_ly as tx

TEP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mau_kiem_thu", "hoi-quy-X.npz")


def australian_co_log() -> np.ndarray:
    bo = tx.nap("australian")
    return tx.zscore(tx.signed_log(bo.X_goc), list(bo.ten_cot))[0]


def main() -> None:
    mau = {ten: tx.nap(ten).X for ten in tx.DANH_SACH}
    mau["australian"] = australian_co_log()
    os.makedirs(os.path.dirname(TEP), exist_ok=True)
    np.savez_compressed(TEP, **mau)
    print(f"Ghi {TEP}: {', '.join(f'{k} {v.shape}' for k, v in mau.items())}")


if __name__ == "__main__":
    main()
