"""
HMI 퍼스널 컬러 진단 시스템 - 메인 애플리케이션
ED-HMI3010-101C 기반 라즈베리 파이 5 통합 시스템
"""

import os
import sys
import cv2
import numpy as np
import threading
import queue
from datetime import datetime

# 환경 설정
os.environ['QT_QPA_PLATFORM'] = 'offscreen'  # HMI 디스플레이 없는 경우

from camera_manager import CameraManager, CameraConfig
from api_server import app, init_database, load_model, save_result
from constants import SEASON_PALETTES, SEASON_CLASSES

# HMI 설정
HMI_WIDTH = 1280
HMI_HEIGHT = 800


class HMIDiagnosticsSystem:
    """HMI 진단 시스템 메인 클래스"""
    
    def __init__(self):
        self.camera = CameraManager()
        self._frame_queue = queue.Queue(maxsize=1)
        self._running = False
        self._display_thread = None
        self._api_server_thread = None
        
        # 결과 저장
        self.last_result = None
        self.session_id = None
        
    def start(self):
        """시스템 시작"""
        print("=" * 60)
        print("HMI 퍼스널 컬러 진단 시스템 시작")
        print("ED-HMI3010-101C 기반")
        print("=" * 60)
        
        # 데이터베이스 초기화
        init_database()
        
        # API 서버 시작
        self._start_api_server()
        
        # 카메라 시작
        if self.camera.start():
            print("✓ 카메라 연결 성공 (4K)")
        else:
            print("✗ 카메라 연결 실패")
        
        # 디스플레이 시작
        self._start_display()
        
        self._running = True
        print("\n시스템 준비 완료!")
        print("F1: 진단 시작 | F2: 결과 보기 | F3: 카메라 미리보기 | ESC: 종료")
    
    def stop(self):
        """시스템 종료"""
        print("\n시스템 종료 중...")
        self._running = False
        
        if self._display_thread:
            self._display_thread.join(timeout=2.0)
        
        self.camera.stop()
        
        print("시스템 종료 완료")
    
    def _start_api_server(self):
        """API 서버 시작 (백그라운드 스레드)"""
        self._api_server_thread = threading.Thread(
            target=lambda: self._run_uvicorn(),
            daemon=True
        )
        self._api_server_thread.start()
        print("✓ API 서버 시작 (http://localhost:8000)")
    
    def _run_uvicorn(self):
        """Uvicorn 서버 실행"""
        import uvicorn
        uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
    
    def _start_display(self):
        """디스플레이 스레드 시작"""
        self._display_thread = threading.Thread(
            target=self._display_loop,
            daemon=True
        )
        self._display_thread.start()
    
    def _display_loop(self):
        """디스플레이 루프"""
        while self._running:
            try:
                frame = self._frame_queue.get_nowait()
                if frame is not None:
                    self._show_frame(frame)
            except queue.Empty:
                pass
            
            # 30 FPS
            import time
            time.sleep(1/30)
        
        # 루프 종료 후
        pass
    
    def _show_frame(self, frame):
        """프레임 표시"""
        # HMI 해상도로 리사이즈
        hmi_frame = cv2.resize(frame, (HMI_WIDTH, HMI_HEIGHT), interpolation=cv2.INTER_AREA)
        
        # 얼굴 분석
        face_result = self.camera.analyze_face(frame)
        
        # 결과 표시
        if face_result["face_detected"]:
            self._draw_face_analysis(frame, face_result)
        
        # 프레임 표시
        cv2.imshow("HMI 퍼스널 컬러 진단", hmi_frame)
        
        # 키 입력 처리
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            self._running = False
        elif key == 70 and ord('1') == 49:  # F1
            self._start_diagnosis()
        elif key == 70 and ord('2') == 50:  # F2
            self._show_results()
        elif key == 70 and ord('3') == 51:  # F3
            self._show_camera_preview()
    
    def _draw_face_analysis(self, frame, face_result):
        """얼굴 분석 결과 그리기"""
        if face_result["landmarks"]:
            # 랜드마크 연결
            landmarks = face_result["landmarks"]
            
            # 얼굴 오вал 그리기
            cv2.polylines(frame, [np.array([landmarks[i] for i in FACE_OVAL_LANDMARKS])], True, (0, 255, 0), 2)
            
            # 양볼 영역 표시
            left_cheek_pts = [landmarks[i] for i in LEFT_CHEEK_LANDMARKS]
            right_cheek_pts = [landmarks[i] for i in RIGHT_CHEEK_LANDMARKS]
            
            cv2.polylines(frame, [np.array(left_cheek_pts)], True, (0, 255, 255), 1)
            cv2.polylines(frame, [np.array(right_cheek_pts)], True, (0, 255, 255), 1)
            
            # 이마 영역 표시
            forehead_pts = [landmarks[i] for i in FOREHEAD_LANDMARKS]
            cv2.polylines(frame, [np.array(forehead_pts)], True, (255, 255, 0), 1)
            
            # 색상 값 표시
            cheek_rgb = face_result.get("cheek_rgb", [0, 0, 0])
            forehead_rgb = face_result.get("forehead_rgb", [0, 0, 0])
            
            cv2.putText(frame, f"양볼 RGB: {cheek_rgb}", (10, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"이마 RGB: {forehead_rgb}", (10, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # 신뢰도 표시
            confidence = face_result.get("face_confidence", 0)
            cv2.putText(frame, f"신뢰도: {confidence:.1%}", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    def _start_diagnosis(self):
        """진단 시작"""
        print("진단 시작...")
        
        # 현재 프레임 사용
        frame = self.camera.get_frame_4k()
        if frame is None:
            print("프레임 없음")
            return
        
        # 얼굴 분석
        face_result = self.camera.analyze_face(frame)
        
        if not face_result["face_detected"]:
            print("얼굴 감지 실패")
            return
        
        # API 에 분석 결과 전송
        session_id = f"sess_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 이미지 파일로 저장
        temp_image_path = os.path.join(os.path.dirname(__file__), "temp_diagnosis.jpg")
        cv2.imwrite(temp_image_path, frame)
        
        # 얼굴 데이터 추출
        cheek_rgb = face_result.get("cheek_rgb", [0, 0, 0])
        cheek_lab = face_result.get("cheek_lab", [0, 0, 0])
        forehead_rgb = face_result.get("forehead_rgb", [0, 0, 0])
        forehead_lab = face_result.get("forehead_lab", [0, 0, 0])
        
        # API 호출
        import requests
        try:
            response = requests.post(
                "http://localhost:8000/analyze",
                files={"file": open(temp_image_path, "rb")},
                data={"session_id": session_id},
                headers={"Content-Type": "multipart/form-data"}
            )
            
            if response.status_code == 200:
                result = response.json()
                self.last_result = result
                self.session_id = session_id
                print(f"✓ 진단 완료: {result['season_korean']} ({result['confidence']:.1%})")
            else:
                print(f"✗ 진단 실패: {response.status_code}")
                
        except Exception as e:
            print(f"✗ 진단 중 오류: {e}")
        
        # 임시 파일 삭제
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
    
    def _show_results(self):
        """결과 표시"""
        if self.last_result:
            print("\n" + "=" * 60)
            print("진단 결과")
            print("=" * 60)
            print(f"Session ID: {self.session_id}")
            print(f"계절: {self.last_result['season_korean']} ({self.last_result['season']})")
            print(f"신뢰도: {self.last_result['confidence']:.1%}")
            print(f"메시지: {self.last_result['message']}")
            print("=" * 60)
        else:
            print("진단 결과가 없습니다.")
    
    def _show_camera_preview(self):
        """카메라 미리보기"""
        frame = self.camera.get_frame_display()
        if frame is not None:
            cv2.imshow("카메라 미리보기", frame)
            cv2.waitKey(0)


# 상수 정의
FACE_OVAL_LANDMARKS = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]

LEFT_CHEEK_LANDMARKS = [50, 101, 118, 117, 123, 187, 207, 206, 205, 36, 93]
RIGHT_CHEEK_LANDMARKS = [280, 330, 347, 346, 352, 411, 427, 426, 425, 266, 323]
FOREHEAD_LANDMARKS = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
    152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
]


def main():
    """메인 함수"""
    system = HMIDiagnosticsSystem()
    
    try:
        system.start()
    except KeyboardInterrupt:
        print("\n사용자가 종료 요청")
    finally:
        system.stop()


if __name__ == "__main__":
    main()