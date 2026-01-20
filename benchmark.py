import time
import requests
import json

# Hàm đo tốc độ
def benchmark_model(model_name, prompt):
    url = "http://localhost:11434/api/generate"
    data = {
        "model": model_name,
        "prompt": prompt,
        "stream": False 
    }
    
    print(f"Dang test toc do mo hinh {model_name}...")
    try:
        start_time = time.time()
        response = requests.post(url, json=data)
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            eval_count = result.get('eval_count', 0)
            duration = result.get('eval_duration', 0) / 1e9 # Đổi ra giây
            
            # Nếu API trả về 0 duration thì tính thủ công
            if duration == 0:
                duration = end_time - start_time
                
            tps = eval_count / duration
            
            print(f"--- KET QUA ---")
            print(f"Tong so token: {eval_count}")
            print(f"Thoi gian chay: {duration:.2f}s")
            print(f"Toc do (Speed): {tps:.2f} tokens/s")
        else:
            print("Loi: Khong ket noi duoc voi Ollama!")
    except Exception as e:
        print(f"Loi xay ra: {e}")

# --- DÒNG QUAN TRỌNG NHẤT Ở ĐÂY ---
if __name__ == "__main__":
    # Đảm bảo bạn đang chạy 'ollama run qwen2:1.5b' ở một cửa sổ khác hoặc background
    benchmark_model("qwen2:1.5b", "Viet mot doan van ngan ve AI trong y te.")