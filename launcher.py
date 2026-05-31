import sys
import subprocess
import re
from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, 
                               QLineEdit, QPushButton, QLabel, QMessageBox)
from PySide6.QtCore import Qt

class OrchestratorLauncher(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Alaz Uzay Havası - Güvenlik Geçidi")
        self.resize(450, 200)
        self.setStyleSheet("background-color: #1e1e1e; color: #ffffff; font-family: Arial;")
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)

        self.info_label = QLabel("Sistemi başlatmak için yetkili e-posta adresini giriniz:")
        self.info_label.setStyleSheet("font-size: 14px; margin-bottom: 10px;")
        layout.addWidget(self.info_label)

        self.email_input = QLineEdit()
        self.email_input.setPlaceholderText("ornek@uzayajansi.gov.tr")
        self.email_input.setStyleSheet("padding: 10px; font-size: 14px; background-color: #2d2d2d; border: 1px solid #444;")
        layout.addWidget(self.email_input)

        self.start_btn = QPushButton("🚀 Çoklu-Ajan Sistemini Başlat")
        self.start_btn.setStyleSheet("padding: 12px; font-size: 14px; font-weight: bold; background-color: #007acc; border-radius: 5px;")
        self.start_btn.clicked.connect(self.verify_and_start)
        layout.addWidget(self.start_btn)

        self.setLayout(layout)

    def is_valid_email(self, email):
        pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
        return re.match(pattern, email)

    def verify_and_start(self):
        email = self.email_input.text().strip()
        
        if not self.is_valid_email(email):
            QMessageBox.critical(self, "Hata", "Geçersiz e-posta formatı. Lütfen tekrar deneyin.")
            return

        self.info_label.setText(f"Yetki onaylandı: {email}\nOrkestratör başlatılıyor...")
        self.start_btn.setEnabled(False)
        self.start_btn.setStyleSheet("background-color: #555555; color: #aaaaaa;")
        
        # Sistemi Subprocess ile başlat (start_system.sh dosyasının çalıştırılabilir izni olmalı)
        try:
            subprocess.Popen(["./start_system.sh"], shell=True)
            QMessageBox.information(self, "Başarılı", "Sistem arka planda başlatıldı. Tarayıcınızda Streamlit açılacaktır.")
            self.close() # Arayüzü kapat, sistem arkada çalışmaya devam etsin
        except Exception as e:
            QMessageBox.critical(self, "Sistem Hatası", f"Başlatma başarısız: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = OrchestratorLauncher()
    window.show()
    sys.exit(app.exec())