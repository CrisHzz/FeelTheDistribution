import http.server
import socketserver
import os

def serve_static_files():
    PORT = 8000
    DIRECTORY = "assets"

    os.chdir(DIRECTORY)

    handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", PORT), handler) as httpd:
        print(f"Serving static files at http://localhost:{PORT}/")
        httpd.serve_forever()

if __name__ == "__main__":
    serve_static_files()