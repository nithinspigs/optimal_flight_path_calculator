import flask

server = flask.Flask(__name__)

@server.route('/')
def main():
    return flask.render_template("optimal_flight_path_calculator.html")
    
@server.route('/plot', methods=['GET'])
def plot():

    args_dict = flask.request.args
    print(args_dict)
    print(args_dict['origin'])
    print(args_dict['dest'])

    return flask.send_file("../../flight/SAN-JFK.jpeg")
    
if __name__ == "__main__":
    server.run()
