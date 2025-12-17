from PySide6.QtWidgets import QWidget, QVBoxLayout, QTabWidget

from .config_camera import CameraConfigPage
from .config_vehicle_types import VehicleTypesConfigPage
from .config_vehicles import VehiclesConfigPage
from .config_fees import FeesConfigPage
from .config_card_reader_page import CardReaderConfigPage
from .config_barrier_page import BarrierConfigPage

class ConfigPage(QWidget):
    """
    Tab Cấu hình:
    - Tab con 'Camera'
    - Tab con 'Loại xe'
    - Tab con 'Xe nội bộ'
    - Tab con 'Phí gửi xe'
    - Tab con 'Đầu đọc thẻ'
    - Tab con 'Barie'
    """
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        tab = QTabWidget(self)
        tab.addTab(CameraConfigPage(parent=self), "Camera")
        tab.addTab(VehicleTypesConfigPage(parent=self), "Loại xe")
        tab.addTab(VehiclesConfigPage(parent=self), "Xe nội bộ")
        tab.addTab(FeesConfigPage(parent=self), "Phí gửi xe") 
        tab.addTab(CardReaderConfigPage(), "Đầu đọc thẻ")
        tab.addTab(BarrierConfigPage(), " Barie ")

        layout.addWidget(tab)
