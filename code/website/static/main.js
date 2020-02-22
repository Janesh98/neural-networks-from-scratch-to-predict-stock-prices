$(document).ready(() => {
    $('#test').on("click", (e) => {
        //let input  = $('#searchText').val();
        //search(input);
        //e.preventDefault();
        console.log("hi");
        $.get("/hello", function(data) {
            console.log(data);
            let train = {
                x : data.trainY,
                y : data.train,
                name : "Train",
                mode : "lines"
            };

            let test = {
                x : data.testY,
                y : data.test,
                name : "Prediction",
                mode : "lines"
            };

            updatePlot([train, test]);
        })
    });
});

// POST
fetch("/hello", {

    // Specify the method
    method: "POST",

    // A JSON payload
    body: JSON.stringify({
        "action": 1
    })
}).then(function (response) { // At this point, Flask has printed our JSON
    let s = response.text();
    console.log(s);
    return s;
})

$.get("/getpythondata", function(data) {
    data = $.parseJSON(data);

    let layout = {
        autosize : true,
        height : 600,
        title : 'Stock Prediction',
        xaxis : {
          title : 'Day',
        },
        yaxis : {
          title : 'Price',
          automargin : true,
        },
    };
    plot(data, layout);
})

function plot(data, layout) {
    console.log("plotting");
    console.log(data);
    data = convertData(data);
    Plotly.react("plot", data, layout);
}

function updatePlot(data) {
    console.log("plotting");
    console.log(data);
    Plotly.addTraces("plot", data);
}

function convertData(stock) {
    let days = [];
    let prices = [];

    for (let i = 0; i < len(stock); i++) {
        prices.push(stock[i]);
        days.push(i);
    }
    
    let data = {
        x : days,
        y : prices,
        name : "Actual",
        mode: "lines"
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