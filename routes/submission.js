var express = require('express');
var router = express.Router();

// GET submission page
router.get('/', function(req,res,next) {
  res.render('submission', {});
});

module.exports = router;
