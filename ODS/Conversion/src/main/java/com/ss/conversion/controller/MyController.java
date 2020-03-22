package com.ss.conversion.controller;

import java.util.Map;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.io.FileSystemResource;
import org.springframework.core.io.Resource;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import com.ss.conversion.model.Authorization;
import com.ss.conversion.service.MyService;

//@CrossOrigin(origins = "*", allowedHeaders = "*")
@RestController
public class MyController {

	@Autowired
	private MyService myService;

	@RequestMapping(value = "/auth", method = RequestMethod.POST, produces = MediaType.APPLICATION_JSON_VALUE)
	public ResponseEntity<?> verifyAndPdfDetails(@RequestBody Authorization auth) {
		ResponseEntity<?> re = null;
		try {
			re = new ResponseEntity<Map<String, Object>>(myService.verifyAndPdfDetails(auth), HttpStatus.OK);
		} catch (Exception e) {
			re = new ResponseEntity<String>(e.getMessage(), HttpStatus.BAD_REQUEST);
		}

		return re;
	}

	@RequestMapping(value = "/getImage", method = RequestMethod.GET)
	public ResponseEntity<?> getImage(@RequestParam String pdfName, @RequestParam int pageNumber) {
		ResponseEntity<?> re = null;
		Resource resource = null;
		HttpHeaders responseHeaders = null;
		try {
			resource = new FileSystemResource(myService.getImage(pdfName, pageNumber));
			responseHeaders = new HttpHeaders();
			responseHeaders.add("Content-Type", MediaType.IMAGE_PNG_VALUE);
			re = new ResponseEntity<Resource>(resource, responseHeaders, HttpStatus.OK);
		} catch (Exception e) {
			re = new ResponseEntity<String>(e.getMessage(), HttpStatus.BAD_REQUEST);
		}

		return re;
	}

	@RequestMapping(value = "/cleanUp", method = RequestMethod.GET)
	public ResponseEntity<?> cleanUp(@RequestParam String pdfName, @RequestParam int totalPages) {
		ResponseEntity<?> re = null;
		try {
			myService.cleanUp(pdfName, totalPages);
			re = new ResponseEntity<>(HttpStatus.OK);
		} catch (Exception e) {
			re = new ResponseEntity<String>(e.getMessage(), HttpStatus.BAD_REQUEST);
		}

		return re;
	}

	// For testing purpose only
	@RequestMapping(value = "/getByteArray", method = RequestMethod.GET)
	public ResponseEntity<?> getByteArray() {
		ResponseEntity<?> re = null;
		try {
			re = new ResponseEntity<byte[]>(myService.getByteArray(), HttpStatus.OK);
		} catch (Exception e) {
			re = new ResponseEntity<String>(e.getMessage(), HttpStatus.BAD_REQUEST);
		}

		return re;
	}

}
