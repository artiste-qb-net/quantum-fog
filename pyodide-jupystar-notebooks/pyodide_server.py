import sys
import socketserver
from http.server import SimpleHTTPRequestHandler


class Handler(SimpleHTTPRequestHandler):
    
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin','*')
        super().end_headers()

if sys.version_info < (3,7,5):
    Handler.extension_map['.wasm'] = 'application/wasm'

if __name__ == '__main__':
    port=8000
    with socketserver.TCPServer(("", port), Handler) as httpd:
        print("Servering at: http://127.0.0.1:()".format(port))
        httpd.serve_forever()
        