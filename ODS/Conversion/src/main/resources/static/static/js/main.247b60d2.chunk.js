(window["webpackJsonpdocument-signer"]=window["webpackJsonpdocument-signer"]||[]).push([[0],{114:function(e,t,n){"use strict";n.r(t);var a=n(0),r=n.n(a),i=n(7),o=n.n(i),s=(n(88),n(32)),c=n(21),u=n(40),l=n(41),d=n(45),g=n(148),m=n(76),p=n(149),h=n(74),f=n(27),D=(n(89),n(140)),E=n(138),x=n(141),S=n(71),O=n(139),y=n(143),v=n(48),w=n.n(v),I=n(51),b=n.n(I),N=n(50),j=n.n(N),U=n(49),R=n.n(U),C=n(68),L=n(25),k=function(e){return{type:"SHOW_APPBAR",value:e}},A=function(e){return{type:"SET_DOCUMENT_INDEX",index:e}},T=function(e,t){return{type:"ADD_OUTPUT",output:e,index:t}},W=function(e){return{type:"SET_NUMBER_OF_DOCUMENTS",number:e}},M=Object(S.a)((function(e){return{root:{flexGrow:1},menuButton:{marginRight:e.spacing(2)},title:{flexGrow:1}}})),_=Object(L.b)((function(e){return{NoOfDocuments:e.NoOfDocuments,DocumentIndex:e.DocumentIndex,SignatureIndex:e.SignatureIndex,ShowAppbar:e.ShowAppbar}}))((function(e){var t=M(),n=function(){var e=document.getElementById("document"),t=e.clientWidth;if(2500===t)return!1;e.style.width=t+100+"px",ee.rerenderDocument()},a=function(){var e=document.getElementById("document"),t=e.clientWidth;if(100===t)return!1;e.style.width=t-100+"px",ee.rerenderDocument()};return e.ShowAppbar?r.a.createElement("div",{className:t.root},r.a.createElement(E.a,{position:"static"},r.a.createElement(O.a,null,r.a.createElement(C.a,{query:"(max-width: 799px)"},(function(i){return i?r.a.createElement(r.a.Fragment,null,r.a.createElement(D.a,{item:!0,container:!0,xs:12,justify:"center"},r.a.createElement(x.a,{onClick:n,color:"inherit"},r.a.createElement(w.a,null)),r.a.createElement(x.a,{onClick:a,color:"inherit",style:{marginRight:"3%"}},r.a.createElement(R.a,null)),r.a.createElement(y.a,{style:{paddingTop:"1%",marginRight:"3%"}},"Pagina: ",e.DocumentIndex+1,"/",e.NoOfDocuments),0!==e.DocumentIndex&&r.a.createElement(x.a,{onClick:function(){e.DocumentIndex>0&&e.dispatch(A(e.DocumentIndex-1))},color:"inherit"},r.a.createElement(j.a,null)),e.DocumentIndex+1!==e.NoOfDocuments&&r.a.createElement(x.a,{onClick:function(){e.DocumentIndex<e.NoOfDocuments-1&&e.dispatch(A(e.DocumentIndex+1))},color:"inherit"},r.a.createElement(b.a,null)))):r.a.createElement(r.a.Fragment,null,r.a.createElement(y.a,{variant:"h6",noWrap:!0,className:t.title},"Firma"),r.a.createElement(x.a,{onClick:n,color:"inherit"},r.a.createElement(w.a,null),r.a.createElement(y.a,null,"\xa0Zoom In")),r.a.createElement(x.a,{onClick:a,color:"inherit",style:{marginRight:"3%"}},r.a.createElement(R.a,null),r.a.createElement(y.a,null,"\xa0Zoom Out")),r.a.createElement(y.a,{style:{marginRight:"3%"}},"Pagina: ",e.DocumentIndex+1,"/",e.NoOfDocuments),0!==e.DocumentIndex&&r.a.createElement(x.a,{onClick:function(){e.DocumentIndex>0&&e.dispatch(A(e.DocumentIndex-1))},color:"inherit"},r.a.createElement(j.a,null),r.a.createElement(y.a,null,"\xa0Pagina Precedente")),e.DocumentIndex+1!==e.NoOfDocuments&&r.a.createElement(x.a,{onClick:function(){e.DocumentIndex<e.NoOfDocuments-1&&e.dispatch(A(e.DocumentIndex+1))},color:"inherit"},r.a.createElement(b.a,null),r.a.createElement(y.a,null,"\xa0Pagina Successiva")))}))))):r.a.createElement(r.a.Fragment,null)})),B=n(147),H=n(12),F=n(144),P=n(145),J=n(146),X=n(150),G=n(117),z=n(30),Z=n.n(z),q=n(31),K=n(73),Y=function(e){function t(e){var n;return Object(s.a)(this,t),(n=Object(u.a)(this,Object(l.a)(t).call(this,e))).state={},n.documents=[],n.defaultSignatures=[],n.nextSignerIndex=1,n.originalDocument=null,n.toggleDrawer=function(e,t){return function(a){(!a||"keydown"!==a.type||"Tab"!==a.key&&"Shift"!==a.key)&&n.setState(Object(H.a)({},e,t))}},n.sideList=function(e){return r.a.createElement("div",{className:n.props.classes.list,role:"presentation",onClick:n.toggleDrawer(e,!1),onKeyDown:n.toggleDrawer(e,!1)},r.a.createElement(F.a,null,r.a.createElement(P.a,null,r.a.createElement(y.a,{style:{textAlign:"center",width:"100%",fontWeight:"bold"}},"Click to Remove Signature")),r.a.createElement(J.a,null),n.props.Output.filter((function(e){return!!e})).map((function(e,t){var a=n.state.signerNames.indexOf(e.user);return r.a.createElement(r.a.Fragment,{key:t},r.a.createElement(P.a,{button:!0,onClick:function(){n.props.dispatch(T(void 0,a)),a===n.props.SignatureIndex&&n.setState({signerMode:!0}),setTimeout((function(){n.getDocument()}),100)}},r.a.createElement(D.a,{container:!0},r.a.createElement(D.a,{item:!0,container:!0,xs:12,justify:"center",alignContent:"center",alignItems:"center"},r.a.createElement("img",{src:n.state.signatureDataArray[a],alt:e.user})),r.a.createElement(D.a,{item:!0,container:!0,xs:12,justify:"center",alignContent:"center",alignItems:"center"},r.a.createElement(y.a,{noWrap:!0},e.user)))),r.a.createElement(J.a,null))}))))},n.componentDidMount=function(){n.getDocument(),document.getElementById("mainBody").onresize=function(){n.rerender()},ee.rerenderDocument=n.rerender},n.getDocument=function(){if(n.setState({progress:!0,showCursor:!1}),n.documents[n.props.DocumentIndex]){var e=n.documents[n.props.DocumentIndex],t=JSON.parse(n.state.signatures)[n.props.SignatureIndex];if("number"===typeof JSON.parse(n.state.signatures)[n.props.SignatureIndex])Z.a.get("http://13.235.145.237:8010/documents/Rectangle/"+n.state.defaultSignature[t],{responseType:"blob"}).then((function(t){var a=window.URL||window.webkitURL;n.props.dispatch(W(n.state.noOfDocuments));var r=document.createElement("img");r.src=a.createObjectURL(e.data),r.onload=function(){n.setState({originalWidth:r.width,originalHeight:r.height})};var i=document.createElement("img");i.src=a.createObjectURL(t.data),i.onload=function(){n.setState({originalSignWidth:i.width,originalSignHeight:i.height})};var o=n.state.signatureDataArray;o[n.props.SignatureIndex]=a.createObjectURL(t.data);var s=a.createObjectURL(e.data);if(n.originalDocument=s,n.props.Output.length>0){var c=[];n.props.Output.forEach((function(e){if(e){var t=n.state.signerNames.indexOf(e.user);n.props.DocumentIndex===e.coordinates.pageNo-1&&c.push({src:o[t],x:e.coordinates.x,y:e.coordinates.y})}})),Object(q.a)([{src:s,x:0,y:0}].concat(c)).then((function(e){s=e,n.setState({documentData:s,signatureData:a.createObjectURL(t.data),progress:!1,signatureDataArray:o})}))}else n.setState({documentData:s,signatureData:a.createObjectURL(t.data),progress:!1,signatureDataArray:o})}));else{var a=t,r=window.URL||window.webkitURL;n.props.dispatch(W(n.state.noOfDocuments));var i=document.createElement("img");i.src=r.createObjectURL(e.data),i.onload=function(){n.setState({originalWidth:i.width,originalHeight:i.height})},n.setState({originalSignWidth:n.state.signatureDimensions[n.props.SignatureIndex].width,originalSignHeight:n.state.signatureDimensions[n.props.SignatureIndex].height});var o=n.state.signatureDataArray;o[n.props.SignatureIndex]=a;var s=r.createObjectURL(e.data);if(n.originalDocument=s,n.props.Output.length>0){var c=[];n.props.Output.forEach((function(e){if(e){var t=n.state.signerNames.indexOf(e.user);n.props.DocumentIndex===e.coordinates.pageNo-1&&c.push({src:o[t],x:e.coordinates.x,y:e.coordinates.y})}})),Object(q.a)([{src:s,x:0,y:0}].concat(c)).then((function(e){s=e,n.setState({documentData:s,signatureData:a,progress:!1,signatureDataArray:o})}))}else n.setState({documentData:s,signatureData:a,progress:!1,signatureDataArray:o})}}else Z.a.get("http://13.235.145.237:8050/getImage?pdfName="+n.state.fileName+"&pageNumber="+n.props.DocumentIndex,{responseType:"blob"}).then((function(e){n.documents[n.props.DocumentIndex]=e;var t=JSON.parse(n.state.signatures)[n.props.SignatureIndex];if("number"===typeof JSON.parse(n.state.signatures)[n.props.SignatureIndex])if(n.defaultSignatures[t]){var a=n.defaultSignatures[t],r=window.URL||window.webkitURL;n.props.dispatch(W(n.state.noOfDocuments));var i=document.createElement("img");i.src=r.createObjectURL(e.data),i.onload=function(){n.setState({originalWidth:i.width,originalHeight:i.height})};var o=document.createElement("img");o.src=r.createObjectURL(a.data),o.onload=function(){n.setState({originalSignWidth:o.width,originalSignHeight:o.height})};var s=n.state.signatureDataArray;s[n.props.SignatureIndex]=r.createObjectURL(a.data);var c=r.createObjectURL(e.data);if(n.originalDocument=c,n.props.Output.length>0){var u=[];n.props.Output.forEach((function(e){if(e){var t=n.state.signerNames.indexOf(e.user);n.props.DocumentIndex===e.coordinates.pageNo-1&&u.push({src:s[t],x:e.coordinates.x,y:e.coordinates.y})}})),Object(q.a)([{src:c,x:0,y:0}].concat(u)).then((function(e){c=e,n.setState({documentData:c,signatureData:r.createObjectURL(a.data),progress:!1,signatureDataArray:s})}))}else n.setState({documentData:c,signatureData:r.createObjectURL(a.data),progress:!1,signatureDataArray:s})}else Z.a.get("http://13.235.145.237:8010/documents/Rectangle/"+n.state.defaultSignature[t],{responseType:"blob"}).then((function(t){var a=window.URL||window.webkitURL;n.props.dispatch(W(n.state.noOfDocuments));var r=document.createElement("img");r.src=a.createObjectURL(e.data),r.onload=function(){n.setState({originalWidth:r.width,originalHeight:r.height})};var i=document.createElement("img");i.src=a.createObjectURL(t.data),i.onload=function(){n.setState({originalSignWidth:i.width,originalSignHeight:i.height})};var o=n.state.signatureDataArray;o[n.props.SignatureIndex]=a.createObjectURL(t.data);var s=a.createObjectURL(e.data);if(n.originalDocument=s,n.props.Output.length>0){var c=[];n.props.Output.forEach((function(e){if(e){var t=n.state.signerNames.indexOf(e.user);n.props.DocumentIndex===e.coordinates.pageNo-1&&c.push({src:o[t],x:e.coordinates.x,y:e.coordinates.y})}})),Object(q.a)([{src:s,x:0,y:0}].concat(c)).then((function(e){s=e,n.setState({documentData:s,signatureData:a.createObjectURL(t.data),progress:!1,signatureDataArray:o})}))}else n.setState({documentData:s,signatureData:a.createObjectURL(t.data),progress:!1,signatureDataArray:o})}));else{var l=t;r=window.URL||window.webkitURL,n.props.dispatch(W(n.state.noOfDocuments));var d=document.createElement("img");d.src=r.createObjectURL(e.data),d.onload=function(){n.setState({originalWidth:d.width,originalHeight:d.height})},n.setState({originalSignWidth:n.state.signatureDimensions[n.props.SignatureIndex].width,originalSignHeight:n.state.signatureDimensions[n.props.SignatureIndex].height});var g=n.state.signatureDataArray;g[n.props.SignatureIndex]=l;var m=r.createObjectURL(e.data);if(n.originalDocument=m,n.props.Output.length>0){var p=[];n.props.Output.forEach((function(e){if(e){var t=n.state.signerNames.indexOf(e.user);n.props.DocumentIndex===e.coordinates.pageNo-1&&p.push({src:g[t],x:e.coordinates.x,y:e.coordinates.y})}})),Object(q.a)([{src:m,x:0,y:0}].concat(p)).then((function(e){m=e,n.setState({documentData:m,signatureData:l,progress:!1,signatureDataArray:g})}))}else n.setState({documentData:m,signatureData:l,progress:!1,signatureDataArray:g})}}))},n.rerender=function(){n.forceUpdate()},n.post=function(){var e=[];return n.props.Output.forEach((function(t){if(t){var a={pageNo:t.coordinates.pageNo,x:72*t.coordinates.x/n.state.dpi,y:72*t.coordinates.y/n.state.dpi},r={user:t.user,coordinates:a};e.push(r)}else e.push(null)})),Z.a.get("http://13.235.145.237:8050/cleanUp?pdfName="+n.state.fileName+"&totalPages="+n.props.NoOfDocuments),window.parent.postMessage(e,"*"),!0},n.state={signerNames:e.payload.signerNames,signatures:JSON.stringify(e.payload.signatures),defaultSignature:e.payload.defaultSignature,noOfDocuments:e.payload.noOfDocuments,noOfSigners:e.payload.noOfSigners,documentData:null,signatureData:null,progress:!1,x:0,y:0,signerMode:!0,outputCoordinates:null,signatureDataArray:e.payload.signatures,name:"Cursor",showCursor:!1,right:!1,fileName:e.payload.fileName,dpi:e.payload.dpi,signatureDimensions:e.payload.signatureDimensions},n}return Object(d.a)(t,e),Object(c.a)(t,[{key:"_onMouseMove",value:function(e){var t=this,n=e.nativeEvent.offsetX,a=e.nativeEvent.offsetY;if(!0===this.state.signerMode){var r=document.getElementById("document").width,i=document.getElementById("document").height,o=[];this.props.Output.forEach((function(e){if(e){var n=t.state.signerNames.indexOf(e.user);t.props.DocumentIndex===e.coordinates.pageNo-1&&e.user!==t.state.signerNames[t.props.SignatureIndex]&&o.push({src:t.state.signatureDataArray[n],x:e.coordinates.x,y:e.coordinates.y})}})),Object(q.a)([{src:this.originalDocument,x:0,y:0}].concat(o,[{src:this.state.signatureData,x:n*(this.state.originalWidth/r),y:a*(this.state.originalHeight/i)}])).then((function(e){var o=t.state.outputCoordinates;o={pageNo:t.props.DocumentIndex+1,x:n*(t.state.originalWidth/r),y:a*(t.state.originalHeight/i)},t.props.dispatch(T({user:t.state.signerNames[t.props.SignatureIndex],coordinates:o},t.props.SignatureIndex)),t.setState({x:n,y:a,documentData:e,outputCoordinates:o,signerMode:!1})}))}}},{key:"componentDidUpdate",value:function(e){e.DocumentIndex===this.props.DocumentIndex&&e.SignatureIndex===this.props.SignatureIndex||this.getDocument()}},{key:"render",value:function(){for(var e=this,t=0,n=(this.props.SignatureIndex+1)%this.state.noOfSigners;n<this.state.noOfSigners;++n)if(!this.props.Output[n]){t=n;break}this.nextSignerIndex=t%this.state.noOfSigners;var a=0;return this.props.Output.forEach((function(e){e&&(a+=1)})),this.state.text?(this.post(),this.props.dispatch(k(!1)),r.a.createElement(r.a.Fragment,null,r.a.createElement(D.a,{container:!0},r.a.createElement(D.a,{container:!0,item:!0,xs:12,justify:"center",alignContent:"center",alignItems:"center"},r.a.createElement(y.a,{variant:"h4",style:{color:"green",marginTop:"5%"}},"Tutte le firme inviate correttamente"))))):r.a.createElement(r.a.Fragment,null,!0===this.state.progress?r.a.createElement(B.a,null):r.a.createElement(D.a,{container:!0},r.a.createElement(D.a,{container:!0,item:!0,xs:12,alignContent:"center",alignItems:"center",justify:"center"},r.a.createElement(D.a,{container:!0,item:!0,xs:12,alignContent:"center",alignItems:"center",justify:"center"},r.a.createElement(D.a,{container:!0,item:!0,xs:6,alignContent:"center",alignItems:"center",justify:"center"},this.props.Output[this.props.SignatureIndex]?r.a.createElement(x.a,{color:"secondary",onClick:function(){e.props.dispatch(T({user:e.state.signerNames[e.props.SignatureIndex],coordinates:e.state.outputCoordinates},e.props.SignatureIndex)),a===e.state.noOfSigners?e.setState({text:!0}):(e.props.dispatch(function(e){return{type:"SET_SIGNATURE_INDEX",index:e}}(e.nextSignerIndex)),e.props.dispatch(A(0)),e.setState({outputCoordinates:null,signerMode:!0,showCursor:!1}))},variant:"contained",fullWidth:!0},a===this.state.noOfSigners?"Salva":"Prossimo Firmatario ( "+this.state.signerNames[this.nextSignerIndex]+" )"):r.a.createElement(y.a,{noWrap:!0},"Firmatario:"," ",this.state.signerNames[this.props.SignatureIndex])),r.a.createElement(D.a,{container:!0,item:!0,xs:6,alignContent:"center",alignItems:"center",justify:"center"},r.a.createElement(x.a,{color:"primary",disabled:!(a>0),onClick:function(t){e.toggleDrawer("right",!0)(t)},variant:"contained",fullWidth:!0},"Elimina Firme"))),r.a.createElement(K.a,{hideScrollbars:!1,className:"scroll-container main"},r.a.createElement("img",{onClick:this._onMouseMove.bind(this),id:"document",alt:"Document",onLoad:function(){e.setState({showCursor:!0})},src:null!==this.state.documentData?this.state.documentData:""})),!0===this.state.showCursor&&r.a.createElement("img",{src:this.state.signatureDataArray[this.props.SignatureIndex],alt:"Cursor",id:this.state.name,className:"cursor",style:{position:"absolute",width:this.state.originalSignWidth&&this.state.signerMode&&document.getElementById("document")?this.state.originalSignWidth*(document.getElementById("document").width/this.state.originalWidth):"0px",height:this.state.originalSignHeight&&this.state.signerMode&&document.getElementById("document")?this.state.originalSignHeight*(document.getElementById("document").height/this.state.originalHeight):"0px",cursor:"none",pointerEvents:"none",display:"none"}}))),r.a.createElement(X.a,{anchor:"right",open:this.state.right,onClose:this.toggleDrawer("right",!1),onOpen:this.toggleDrawer("right",!0)},this.sideList("right")))}}]),t}(r.a.Component),$=Object(L.b)((function(e){return{DocumentIndex:e.DocumentIndex,SignatureIndex:e.SignatureIndex,Output:e.Output,NoOfDocuments:e.NoOfDocuments}}))(Object(G.a)({list:{width:250},fullList:{width:"auto"}})(Y)),Q=Object(m.a)({palette:{primary:{main:"#d8d8d8"},secondary:{main:"#f0c030"}}}),V=function(e){function t(){var e,n;Object(s.a)(this,t);for(var a=arguments.length,r=new Array(a),i=0;i<a;i++)r[i]=arguments[i];return(n=Object(u.a)(this,(e=Object(l.a)(t)).call.apply(e,[this].concat(r)))).state={progress:!0,payload:{noOfSigners:0,signerNames:[],signatures:[],defaultSignature:[],noOfDocuments:1,fileName:"",dpi:0,signatureDimensions:[],showDocument:!0}},n.signatures=[],n.imageToDataUri=function(e,t,a,r){var i=document.createElement("canvas"),o=i.getContext("2d");i.width=t,i.height=a;var s=new Image;s.src=e,s.onload=function(){o.drawImage(s,0,0,t,a),n.signatures[r]=i.toDataURL()}},n.receiveMessage=function(e){var t=JSON.parse(e.data),a={documentId:t.documentId,timestamp:t.timestamp,token:t.token};Z.a.post("http://13.235.145.237:8050/auth",a).then((function(e){var a=[],r=0,i=[];t.signers.forEach((function(e,t){a.push(e.name),e.signature?(n.imageToDataUri("data:image/png;base64, "+e.signature,e.width,e.height,t),i.push({height:e.height,width:e.width})):(i.push(null),n.signatures[t]=r,r+=1)})),setTimeout((function(){var r={noOfSigners:t.signers.length,signerNames:a,signatures:n.signatures,defaultSignature:e.data.rectangles,noOfDocuments:e.data.totalPages,fileName:e.data.filename,dpi:e.data.dpi,signatureDimensions:i};n.setState({payload:r,progress:!1,showDocument:!0})}),2e3)})).catch((function(e){n.setState({progress:!1,showDocument:!1}),n.props.dispatch(k(!1))}))},n.componentDidMount=function(){n.setState({progress:!0}),window.addEventListener("message",n.receiveMessage,!1),window.parent.postMessage(!0,"*")},n}return Object(d.a)(t,e),Object(c.a)(t,[{key:"render",value:function(){var e=this;return r.a.createElement(r.a.Fragment,null,this.state.progress?r.a.createElement(B.a,null):r.a.createElement(r.a.Fragment,null,r.a.createElement(g.a,null),r.a.createElement(h.a,null,r.a.createElement(p.a,{theme:Q},r.a.createElement(f.c,null,r.a.createElement(f.a,{path:"/**",component:function(){return r.a.createElement(r.a.Fragment,null,r.a.createElement(_,null),e.state.showDocument?r.a.createElement($,{payload:e.state.payload}):r.a.createElement(r.a.Fragment,null,r.a.createElement(D.a,{container:!0},r.a.createElement(D.a,{container:!0,item:!0,xs:12,justify:"center",alignContent:"center",alignItems:"center"},r.a.createElement(y.a,{variant:"h4",style:{color:"red",marginTop:"5%"}},"Autenticazione fallita")))))}}))))))}}]),t}(r.a.Component);V.rerenderDocument=null;var ee=Object(L.b)()(V);Boolean("localhost"===window.location.hostname||"[::1]"===window.location.hostname||window.location.hostname.match(/^127(?:\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)){3}$/));var te=n(33),ne=function(){var e=arguments.length>0&&void 0!==arguments[0]?arguments[0]:1,t=arguments.length>1?arguments[1]:void 0;switch(t.type){case"SET_NUMBER_OF_DOCUMENTS":return t.number;default:return e}},ae=function(){var e=arguments.length>0&&void 0!==arguments[0]?arguments[0]:0,t=arguments.length>1?arguments[1]:void 0;switch(t.type){case"SET_DOCUMENT_INDEX":return t.index;default:return e}},re=function(){var e=arguments.length>0&&void 0!==arguments[0]?arguments[0]:0,t=arguments.length>1?arguments[1]:void 0;switch(t.type){case"SET_SIGNATURE_INDEX":return t.index;default:return e}},ie=n(26),oe=function(){var e=arguments.length>0&&void 0!==arguments[0]?arguments[0]:[],t=arguments.length>1?arguments[1]:void 0;switch(t.type){case"ADD_OUTPUT":var n=Object(ie.a)(e);return n[t.index]=t.output,n;default:return e}},se=function(){var e=!(arguments.length>0&&void 0!==arguments[0])||arguments[0],t=arguments.length>1?arguments[1]:void 0;switch(t.type){case"SHOW_APPBAR":return t.value;default:return e}},ce=Object(te.b)({NoOfDocuments:ne,DocumentIndex:ae,SignatureIndex:re,Output:oe,ShowAppbar:se}),ue=Object(te.c)(ce);o.a.render(r.a.createElement(L.a,{store:ue},r.a.createElement(ee,null)),document.getElementById("root")),"serviceWorker"in navigator&&navigator.serviceWorker.ready.then((function(e){e.unregister()}))},83:function(e,t,n){e.exports=n(114)},88:function(e,t,n){},89:function(e,t,n){}},[[83,1,2]]]);