
var ctx = document.getElementById('rocChart').getContext('2d');
var chart = new Chart(ctx, {
    type: 'line',
    data: {
        labels: fpr,
        datasets: [
            {
                label: 'ROC curve (area = ' + roc_auc.toFixed(2) + ')',
                borderColor: 'rgb(186, 65, 5)',
                borderWidth: 2,
                data: tpr,
                fill: false
            },
            {
                label: 'Randomly guessed curves',
                borderColor: 'grey',
                borderWidth: 2,
                data: fpr,
                fill: false,
                borderDash: [5, 5]
            }
        ]
    },
    options: {
        scales: {
            x: {
                type: 'linear',
                position: 'bottom',
                title: {
                    display: true,
                    text: 'False Positive Rate'
                }
            },
            y: {
                type: 'linear',
                position: 'left',
                title: {
                    display: true,
                    text: 'True Positive Rate'
                }
            }
        },
        title: {
            display: true,
            text: 'Receiver Operating Characteristic (ROC)'
        },
        plugins: {
            legend: {
                position: 'bottom'
            }
        }
    }
});
