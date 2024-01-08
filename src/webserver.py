import flask
import os
import make_path_function

server = flask.Flask(__name__)
server.root_path = "/Users/nithin/Personal/git-repos/optimal_flight_path_calculator/"
print(server.root_path)
#os.path.join(server.root_path, "/flight/SAN-JFK.jpeg")

@server.route('/')
def main():
    return flask.render_template("optimal_flight_path_calculator.html")
    
@server.route('/plot', methods=['GET'])
def plot():

    args_dict = flask.request.args
    #print(args_dict)
    #print(args_dict['origin'])
    #print(args_dict['dest'])
    
    uri = "static/" + args_dict['origin'] + "-" + args_dict['dest'] + ".jpeg"
    if(os.path.isfile(uri) == False):
        if(make_path_function.process_results(args_dict['origin'], args_dict['dest'])):
            uri = "error"

    #make response based on origin and dest
    #first need to run make_path.py to create the image
    #uri = flask.url_for('static', filename='SAN-JFK.jpeg')
    #response = {'uri': os.path.join(server.root_path, "flight/SAN-JFK.jpeg")}
    response = {'uri': uri}
    print(response)
    return flask.jsonify(response)
    #return flask.send_file("../../flight/SAN-JFK.jpeg")
    
if __name__ == "__main__":
    server.run()
