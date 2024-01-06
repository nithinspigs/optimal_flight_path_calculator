function display_input(){
    $("#origin_dest_input").empty();
    var origin_dest_input_template = $("<div class = row></div>");
    var origin_template = $("<input type=text id=origin name=origin>");
    var dest_template = $("<input type=text id=dest name=dest>");
    var submit_button = $("<button class = submit-button>submit</button>");
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

function display_plot(img){
  //display_input()
  $("#plot").empty();
  var reader  = new FileReader();
  reader.onload = function(e)  {
    var image = document.createElement("img");
    // the result image data
    image.src = e.target.result;
    $("#plot").append(image);
  }
  // you have to declare the file loading
  reader.readAsDataURL(img);
}

function new_plot(origin, dest){
  console.log(origin)
  console.log(dest)
  var origin_dest = {"origin": origin, "dest": dest}
  $.ajax({
    type: "GET",
    url: "plot",
    dataType: "jpeg",
    contentType: "application/json; charset=utf-8",
    data: origin_dest,
    // result is what server sends back to client upon success
    success: function(result){
        display_plot(result)
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
