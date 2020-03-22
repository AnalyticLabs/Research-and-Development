$(function() {
  $("#testarea")
    .mousemove(function(e) {
      $(".cursor")
        .show()
        .css({
          left: e.clientX,
          top: e.clientY + $("#testarea").scrollTop(),
          display: "block"
        });
    })
    .mouseout(function() {
      $(".cursor").hide();
    });
});
