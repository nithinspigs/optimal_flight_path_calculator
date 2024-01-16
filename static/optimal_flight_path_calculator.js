function display_input(){
    $("#origin_dest_input").empty();
    var origin_dest_input_template = $("<div class = row></div>");
    var origin_template = $("<input type=text id=origin name=origin><p>");
    var dest_template = $("<input type=text id=dest name=dest><p>");
    var submit_button = $("<button id=submit-button>submit</button>");
    $(submit_button).click(function(){
        new_plot(document.getElementById('origin').value, document.getElementById('dest').value);
    });
    origin_dest_input_template.append("Origin: ");
    origin_dest_input_template.append(origin_template);
    origin_dest_input_template.append(" Destination: ");
    origin_dest_input_template.append(dest_template);
    origin_dest_input_template.append(submit_button);
    $("#origin_dest_input").append(origin_dest_input_template);
    console.log(origin_dest_input_template)
}

function display_plot(response){
    //display_input()
    $("#plot").empty();
    //var plot_template = $("<div class = row></div>");
    var plot_template = '<img id="img" src="' + response['uri'] + '" alt="plot"></img>';
    $("#plot").append(plot_template);
    console.log(plot_template)
}

function display_msg(msg, element_id){
    $(element_id).empty();
    var msg_template = $("<div>" + msg + "</div>");
    $(element_id).append(msg_template);
    console.log(msg_template)
}

function new_plot(origin, dest){
    console.log(origin)
    console.log(dest)
    display_msg("Generating plot…", "#msg")
    display_msg("", "#error")
    display_msg("", "#plot")
    var origin_dest = {"origin": origin, "dest": dest}
    $.ajax({
        type: "GET",
        url: "plot",
        dataType: "json",
        contentType: "application/json; charset=utf-8",
        data: origin_dest,
        // result is what server sends back to client upon success
        success: function(response){
            console.log(response)
            if(response['uri'].localeCompare("error") == 0) {
                display_msg("", "#msg")
                display_msg("Origin or destination error", "#error")
                display_msg("", "#plot")
            } else {
                display_msg("", "#msg")
                display_msg("", "#error")
                display_plot(response)
            }
        },
        error: function(request, status, error){
            console.log("Error");
            console.log(request)
            console.log(status)
            console.log(error)
        }
    });
}

$(document).ready(function(){
    display_input();
})
