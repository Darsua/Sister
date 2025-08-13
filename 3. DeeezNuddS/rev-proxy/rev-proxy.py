from http.server import BaseHTTPRequestHandler, HTTPServer
import requests
import os

os.system('clear')
print("Attempting to start reverse proxy...\n")

TARGET_HOST = "192.168.100.20"
TARGET_PORT = 8080

class ProxyHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        print("\n[*]Request from ", end="")

        url = f"http://{TARGET_HOST}:{TARGET_PORT}{self.path}"
        headers = dict(self.headers)
        headers['Accept-Encoding'] = "identity"
        
        try:
            r = requests.get(url, headers=headers)
            self.send_response(r.status_code)

            for key, value in r.headers.items():
                self.send_header(key, value)
            self.end_headers()

            self.wfile.write(r.content)

        except requests.ConnectionError:
            self.send_error(503, "Service Unavailable")

def run_proxy():
    server_addr = ('', 80)
    httpd = HTTPServer(server_addr, ProxyHandler)

    print("Reverse proxy running on port 80...")
    print("Long live Gigi Murin.")

    httpd.serve_forever()

if __name__ == '__main__':
    run_proxy()
