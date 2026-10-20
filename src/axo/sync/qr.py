"""二维码渲染：只做 SVG 字节，Qt 用 QSvgRenderer 画出来。

选 SVG 而不是 PNG：PySide6 自带 QtSvg，不需要 Pillow；二维码是矢量，缩放不糊。
"""

from __future__ import annotations

__all__ = ["make_qr_svg"]


def make_qr_svg(text: str) -> bytes:
    import qrcode
    from qrcode.image.svg import SvgPathImage

    qr = qrcode.QRCode(
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=8,
        border=2,
    )
    qr.add_data(text)
    qr.make(fit=True)
    image = qr.make_image(image_factory=SvgPathImage)
    data = image.to_string()
    return data if isinstance(data, bytes) else data.encode("utf-8")
