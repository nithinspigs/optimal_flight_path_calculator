import flask

server = flask.Flask(__name__)

@server.route('/')
def hello_world():
    return "Hello World!"
    
@server.route('/slay')
def slay():
    return flask.send_file("../../flight/SAN-JFK.jpeg")
    
if __name__ == "__main__":
    server.run()
