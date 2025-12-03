# x, y, w, h = bbox
            
#             # Görüntü sınırlarını taşmamak için önlem (Clip)
#             img_h, img_w, _ = cv_image.shape
#             x = max(0, min(x, img_w))
#             y = max(0, min(y, img_h))
#             w = max(1, min(w, img_w - x)) # Genişlik en az 1 olsun
#             h = max(1, min(h, img_h - y)) # Yükseklik en az 1 olsun
            
#             # 1. Kutunun içindeki resmi kesip al (ROI - Region of Interest)
#             roi = cv_image[y:y+h, x:x+w]
            
#             # 2. İstatistikleri hesapla
#             # Griye çevir (Tek kanalda işlem yapmak kolaydır)
#             if roi.size > 0:
#                 gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                
#                 # Ortalama Parlaklık (Mean) ve Varyans (StdDev)
#                 mean_val = np.mean(gray_roi)
#                 std_dev = np.std(gray_roi)
                

#                 # 3. KARANLIK TESTİ (Kamerayı kapatınca)
#                 if mean_val < 10: # Simsiyahsa (Eşik: 10)
#                     self.get_logger().warn(" görüntü karanlık. Takip Bırakıldı.")
#                     self.dark_frame_count += 1
#                     if self.dark_frame_count >= 5 :
#                         self.dark_frame_count = 0
#                         self.takip_modu = False
#                         return
#                 else:
#                     self.dark_frame_count = 0

#                 # 4. DÜZLÜK TESTİ (Duvar/Boşluk)
#                 if std_dev < 10: # Görüntüde hiç detay yoksa, dümdüz renksizse
                    
#                     self.get_logger().warn("düz şeyler takip ediliyor. Takip Bırakıldı.")
#                     self.flat_frame_count +=1
#                     if self.flat_frame_count >= 5 :
#                         self.flat_frame_count = 0
#                         self.takip_modu = False
#                         return
#                 else:
#                     self.dark_frame_count = 0
#             # ---------------------------------------------