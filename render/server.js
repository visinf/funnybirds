var express = require('express');
var app = express();
app.engine('html', require('ejs').renderFile);
app.set('views', __dirname);
//console.log(__dirname + '/views')
app.set('view engine', 'html');

var fs = require("fs");
const puppeteer = require('puppeteer')

app.get('/render', function (req, res) {

    var params = '?' + req.url.split('?')[1];
    console.log(params)
    ; (async () => {
        const browser = await puppeteer.launch( {
            headless: ! process.env.VISIBLE,
            args: [
                '--use-gl=swiftshader',
                '--no-sandbox',
                '--enable-surface-synchronization'
            ]
        } )
        
        const page = await browser.newPage()

        await page.setViewport({ width: 1256, height: 1256 })
        await page.goto('http://localhost:8081/page' + params)
        await new Promise(resolve => setTimeout(resolve, 1000));
        x = await page.screenshot({ path: 'my_screenshot.png' , encoding:'base64'})
        await browser.close()
        res.end( x );
        
    })()
})

app.get('/page', function (req, res) {

    var render_mode = req.query.render_mode

    var camera_distance = req.query.camera_distance
    var camera_pitch = req.query.camera_pitch
    var camera_roll = req.query.camera_roll

    var light_distance = req.query.light_distance
    var light_pitch = req.query.light_pitch
    var light_roll = req.query.light_roll

    var beak_model = req.query.beak_model
    var beak_color = req.query.beak_color

    var foot_model = req.query.foot_model

    var eye_model = req.query.eye_model

    var tail_model = req.query.tail_model
    var tail_color = req.query.tail_color

    var wing_model = req.query.wing_model
    var wing_color = req.query.wing_color

    var bg_objects = req.query.bg_objects
    var bg_radius = req.query.bg_radius
    var bg_pitch = req.query.bg_pitch
    var bg_roll = req.query.bg_roll
    var bg_scale_x = req.query.bg_scale_x
    var bg_scale_y = req.query.bg_scale_y
    var bg_scale_z = req.query.bg_scale_z
    var bg_rot_x = req.query.bg_rot_x
    var bg_rot_y = req.query.bg_rot_y
    var bg_rot_z = req.query.bg_rot_z
    var bg_color = req.query.bg_color

    console.log(beak_model)
    console.log(beak_color)
    //res.send("<html> <head>server Response</head><body><h1> This page was render direcly from the server <p>Hello there welcome to my website</p></h1></body></html>");
    app.use(express.static(__dirname + '/js/three.js-master/examples'));

    res.render('./page.html', {render_mode: render_mode, camera_distance: camera_distance, camera_pitch: camera_pitch, camera_roll: camera_roll, light_distance:light_distance, light_pitch: light_pitch, light_roll: light_roll, beak_model: beak_model, beak_color: beak_color, foot_model: foot_model, eye_model: eye_model, tail_model: tail_model, tail_color:tail_color, wing_model: wing_model, wing_color: wing_color, bg_objects: bg_objects, bg_scale_x:bg_scale_x, bg_scale_y:bg_scale_y, bg_scale_z:bg_scale_z, bg_rot_x:bg_rot_x, bg_rot_y:bg_rot_y, bg_rot_z:bg_rot_z, bg_color:bg_color, bg_radius:bg_radius, bg_pitch:bg_pitch, bg_roll:bg_roll});
})


var server = app.listen(8081, function () {
   var host = server.address().address
   var port = server.address().port
   console.log("Example app listening at http://%s:%s", host, port)
})


