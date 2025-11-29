import socket
import json
from pathlib import Path

HOST = '127.0.0.1'  # Localhost
PORT = 5000         # Same port used in Unity Tetris game
OUTPUT_FILE = Path(__file__).parent / "assets/tetris_state.json"

def start_server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        print(f"Server listening on {HOST}:{PORT}")
        conn, addr = s.accept()
        with conn:
            print(f"Connected by {addr}")
            data_buffer = ""
            while True:
                data = conn.recv(4096).decode('utf-8')
                if not data:
                    break

                data_buffer += data
                # Messages are separated by newline
                while '\n' in data_buffer:
                    message, data_buffer = data_buffer.split('\n', 1)
                    try:
                        payload = json.loads(message)
                        grid = payload.get("grid")
                        lines_cleared = payload.get("linesCleared")
                        
                        print("Received grid:")
                        # Pretty print the grid array on same line
                        print(json.dumps(grid))
                        print("Received lines cleared:")
                        print(lines_cleared)
                        
                        print("-----")

                        # Save to JSON file
                        '''
                        with open(OUTPUT_FILE, "w") as f:
                            json.dump(grid, f)
                        print(f"✅ Grid saved to {OUTPUT_FILE}")
                        '''

                    except json.JSONDecodeError:
                        print("Received invalid JSON message (skipped).")
                    except KeyError:
                        print("JSON missing 'data' field (skipped).")

if __name__ == "__main__":
    start_server()