let dates = {"start" : "2019-01-01",
             "end" : "2019-12-31"};

function handle_date(date, start=true) {
    // date is string in format y-m-d
    if (start) {
        dates["start"] = date;
    }

    else {
        dates["end"] = date;
    }
    console.log(dates);
}

$(document).ready(() => {
    $('#searchForm').on('submit', (e) => {
        let stock  = $('#searchText').val();
        //console.log(stock);
        e.preventDefault();

        // POST
        fetch("/hello", {
            method: "POST",
            body: JSON.stringify({
                "stock": stock,
                "startDate" : dates["start"],
                "endDate" : dates["end"]
            })
        }).then(function (response) {
            // get response from flask
            return response.json()

        }).then(function (data) {
            // plot new data
            console.log(data);
            let title = data.stock;

            let actual = {
                x : data.actualX,
                y : data.actual,
                name : "actual",
                mode : "lines"
            };

            let train = {
                x : data.trainX,
                y : data.train,
                name : "Train",
                mode : "lines"
            };

            let test = {
                x : data.testX,
                y : data.test,
                name : "Prediction",
                mode : "lines"
            };

            plot([actual, train, test], title, convert = false);

        })
    });
});

$.get("/getpythondata", function(data) {
    data = $.parseJSON(data);
    plot(data);
})

function plot(data, title="Stock Prediction", convert=true) {
    let layout = {
        autosize : true,
        height : 600,
        title : title,
        xaxis : {
          title : "Day",
        },
        yaxis : {
          title : "Price",
          automargin : true,
        },
    };
    //console.log("plotting");
    //console.log(data);
    if (convert) {
        data = convertData(data);
    }
    Plotly.react("plot", data, layout);
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