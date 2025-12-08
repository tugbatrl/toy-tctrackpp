#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray 
from cv_bridge import CvBridge, CvBridgeError
import cv2
import torch
import numpy as np
import sys
import os
import time
from ultralytics import YOLO as yolo

# --- DOSYA YOLLARI ---
# Burayı senin son attığın yola göre ayarladım. 
# Eğer hata verirse eski yolunu ('/home/tugba/Toy-iha/...') kullan.
current_dir = '/home/tugba/Desktop/MyDocuments/Toy-iha/iha_ws/src/takip_sistemi/takip_sistemi'


sys.path.append(current_dir)

from pysot.core.config import cfg
from pysot.models.utile_tctrackplus.model_builder import ModelBuilder_tctrackplus
from pysot.tracker.tctrackplus_tracker import TCTrackplusTracker
from pysot.utils.model_load import load_pretrain

class TakipciDugumu(Node):
    def __init__(self):
        super().__init__('tracker_node')
        self.get_logger().info("🦁 TCTrack++ Takip Düğümü Başlatılıyor... Sakin ol, hallediyoruz!")

        # 1. ARAÇLARI HAZIRLA
        self.bridge = CvBridge()
        self.tracker = None
        self.takip_modu = False 
        
        # 2. AYARLAR
        self.KAYIP_ESIGI = 0.80      # Biraz düşürdüm ki hemen pes etmesin
        self.KILITLENME_SURESI = 4.0 # 4 Saniye kuralı
        self.TOLERANS_SURESI = 1.0   # 1 Saniye tolerans
        
        # SAYAÇLAR
        self.dark_frame_count = 0
        self.flat_frame_count = 0
        self.locked_start = None     # Kilitlenme başlangıcı
        self.last_seen_time = 0      # Son görülme zamanı
        self.this_time = 0           # Şu anki zaman

        # 3. MODELLERİ YÜKLE
        self.init_tctrack()
        self.init_yolo()

        # 4. YAYINCILAR VE ABONE
        self.bbox_pub = self.create_publisher(Float32MultiArray, '/tracker/bbox', 10)
        self.debug_pub = self.create_publisher(Image, '/tracker/debug_image', 10)
        self.subscription = self.create_subscription(Image, '/camera/image_raw', self.resim_geldi_callback, 10)
        
        self.get_logger().info("✅ Sistem Hazır! Gönder gelsin...")

    def init_yolo(self):
        try:
            # Model yolunu garantiye alalım
            yolo_path = os.path.join(current_dir, "snapshot", "best.pt")
                
            self.detector = yolo(yolo_path)
            self.get_logger().info("🕵️ Dedektif (YOLO) Hazır!")
        except Exception as e:
            self.get_logger().error(f"YOLO Hatası: {e}")

    def init_tctrack(self):
        try:
            config_path = os.path.join(current_dir, 'experiments', 'TCTrack', 'config.yaml')
            snapshot_path = os.path.join(current_dir, 'snapshot', 'TCTrack.pth')

            cfg.merge_from_file(config_path)
            cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
            device = torch.device('cuda' if cfg.CUDA else 'cpu')

            model = ModelBuilder_tctrackplus('test')
            model = load_pretrain(model, snapshot_path).eval().to(device)
            self.tracker = TCTrackplusTracker(model)
            self.hp = [cfg.TRACK.PENALTY_K, cfg.TRACK.WINDOW_INFLUENCE, cfg.TRACK.LR]

            self.get_logger().info(f"✅ TCTrack++ Yüklendi! Cihaz: {device}")
        except Exception as e:
            self.get_logger().error(f"❌ TCTrack Hatası: {e}")

    def resim_geldi_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            return

        # --- ZAMANI GÜNCELLE (EN ÖNEMLİ KISIM) ---
        self.this_time = time.time()

        # ---------------------------------------------------------
        # DURUM 1: ARAMA MODU (YOLO)
        # ---------------------------------------------------------
        if not self.takip_modu:
            results = self.detector(cv_image, verbose=False)
            
            for result in results:
                for box in result.boxes:
                    conf = float(box.conf[0])
                    
                    if conf > 0.60: # %60'tan eminse
                        # Koordinatları al
                        coords = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = map(int, coords)
                        
                        w = x2 - x1
                        h = y2 - y1
                        
                        # AŞIRI BÜYÜK HEDEF KONTROLÜ (Ekranın %90'ı ise alma)
                        h_img, w_img, _ = cv_image.shape
                        if w > w_img * 0.9: continue

                        # TCTrack Başlat
                        init_rect = (x1, y1, w, h)
                        self.tracker.init(cv_image, init_rect)
                        
                        # DURUMLARI GÜNCELLE
                        self.takip_modu = True
                        self.locked_start = self.this_time   # Kronometre Başladı!
                        self.last_seen_time = self.this_time # Şimdi gördüm!
                        
                        self.get_logger().info("🚀 HEDEF BULUNDU! Sayaç Başlıyor...")
                        
                        # İlk kareyi hemen çiz (Kullanıcı görsün)
                        cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        cv2.putText(cv_image, "LOCKED", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        break 
                if self.takip_modu: break

        # ---------------------------------------------------------
        # DURUM 2: TAKİP MODU (TCTrack++)
        # ---------------------------------------------------------
        else:
            outputs = self.tracker.track(cv_image, self.hp)
            score = outputs['best_score']
            bbox = list(map(int, outputs['bbox']))
            
            # --- GÜVENLİK FİLTRELERİ (Karanlık / Düzlük) ---
            x, y, w, h = bbox
            img_h, img_w, _ = cv_image.shape
            
            # Sınır Kontrolü
            x = max(0, min(x, img_w-1)); y = max(0, min(y, img_h-1))
            w = max(1, min(w, img_w - x)); h = max(1, min(h, img_h - y))
            
            roi = cv_image[y:y+h, x:x+w]
            
            if roi.size > 0:
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                mean_val = np.mean(gray_roi)
                std_dev = np.std(gray_roi)

                # Karanlık Testi
                if mean_val < 10: 
                    self.dark_frame_count += 1
                    if self.dark_frame_count >= 5:
                        self.takip_modu = False; self.dark_frame_count = 0
                        self.get_logger().warn("🌑 ORTAM KARANLIK - Takip Bitti"); return
                else: self.dark_frame_count = 0

                # Düzlük Testi
                if std_dev < 10:
                    self.flat_frame_count += 1
                    if self.flat_frame_count >= 5:
                        self.takip_modu = False; self.flat_frame_count = 0
                        self.get_logger().warn("🧱 HEDEF DÜZ DUVAR - Takip Bitti"); return
                else: self.flat_frame_count = 0

            # ---------------------------------------------------------
            # KRONOMETRE VE TOLERANS MANTIĞI (DÜZELTİLEN KISIM)
            # ---------------------------------------------------------
            
            # 1. HEDEFİ GÖRÜYORSAK (SKOR İYİ)
            if score >= self.KAYIP_ESIGI:
                # KRİTİK DÜZELTME: Gördüğümüz için zamanı güncelle!
                self.last_seen_time = self.this_time 
                
                # Eğer sayaç resetlendiyse tekrar başlat
                if self.locked_start is None:
                    self.locked_start = self.this_time

                # Geçen Süreyi Hesapla
                gecen_sure = self.this_time - self.locked_start
                
                # Ekrana Yaz
                cv2.putText(cv_image, f"SURE: {gecen_sure:.1f}s", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                
                # ÇİZİM (Kırmızı Kutu)
                p1 = (bbox[0], bbox[1])
                p2 = (bbox[0] + bbox[2], bbox[1] + bbox[3])
                cv2.rectangle(cv_image, p1, p2, (0, 0, 255), 3)
                cv2.putText(cv_image, f"TRACKING ({score:.2f})", (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # 4 SANİYE KONTROLÜ (ZAFER ANII!!!)
                if gecen_sure >= self.KILITLENME_SURESI:
                    # Kalın ve Büyük Yazı
                    cv2.putText(cv_image, "LOCKED SUCCESSFULLY", (bbox[0], bbox[1]-40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
                    self.get_logger().info("🏆 KİLİTLENME BAŞARILI!")

            # 2. HEDEF KAYIPSA (SKOR DÜŞÜK)
            else:
                # Ne kadar zamandır kayıp?
                kayip_suresi = self.this_time - self.last_seen_time
                
                if kayip_suresi < self.TOLERANS_SURESI:
                    # Tolerans içindeyiz, SABRET
                    cv2.putText(cv_image, "KAYIP - BEKLENIYOR...", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    self.get_logger().info(f"⚠️ Hedef Kayıp... ({kayip_suresi:.1f}s)")
                else:
                    # Tolerans doldu, BİTİR.
                    self.takip_modu = False
                    self.locked_start = None
                    self.get_logger().warn("🚫 HEDEF KAÇTI! Başa Dönülüyor...")
                    return

            # SONUÇLARI YAYINLA
            bbox_msg = Float32MultiArray()
            bbox_msg.data = [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
            self.bbox_pub.publish(bbox_msg)

        # GÖRÜNTÜYÜ HER ZAMAN YAYINLA
        debug_msg = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
        self.debug_pub.publish(debug_msg)
#noluor
def main(args=None):
    rclpy.init(args=args)
    node = TakipciDugumu()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()