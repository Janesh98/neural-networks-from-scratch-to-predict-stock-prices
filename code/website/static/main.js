$(document).ready(() => {
    $('#test').on('click', (e) => {
        //let input  = $('#searchText').val();
        //search(input);
        //e.preventDefault();
        console.log("hi");
        $.get("/hello", function(data) {
            data = $.parseJSON(data);
            //console.log(data);
            plot(data);
        })
    });
});

// POST
fetch('/hello', {

    // Specify the method
    method: 'POST',

    // A JSON payload
    body: JSON.stringify({
        "greeting": "Hello from the browser!"
    })
}).then(function (response) { // At this point, Flask has printed our JSON
    let s = response.text();
    console.log(s);
    return s;
})

$.get("/getpythondata", function(data) {
    data = $.parseJSON(data);
    plot(data);
})

function plot(data) {
    console.log("plotting");
    console.log(data);
    data = convertData(data);
    Plotly.newPlot('plot', data);
}

function convertData(stock) {
    let days = [];
    let prices = [];

    for (let i = 0; i < len(stock); i++) {
        prices.push(stock[i]);
        days.push(i);
    }
    
    let data = {
        x: days,
        y: prices,
        mode: 'lines'
      };

    // data is a dictionary within a list [{}]
    return [data];
}

function len(d) {
    let length = 0;
    for (i in d) {
        length += 1;
    }

    return length;
}