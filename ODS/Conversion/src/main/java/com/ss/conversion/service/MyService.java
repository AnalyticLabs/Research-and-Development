package com.ss.conversion.service;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import javax.crypto.Mac;
import javax.crypto.spec.SecretKeySpec;

import org.apache.commons.codec.binary.Hex;
import org.apache.commons.io.FilenameUtils;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.rendering.ImageType;
import org.apache.pdfbox.rendering.PDFRenderer;
import org.apache.pdfbox.tools.imageio.ImageIOUtil;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpMethod;
import org.springframework.stereotype.Service;
import org.springframework.util.StreamUtils;
import org.springframework.web.client.RestTemplate;

import com.ss.conversion.model.Authorization;

@Service
public class MyService {

	@Value("${temp.directory}")
	private String tempDirectory;

	@Value("${pdf.download.url}")
	private String pdfDownloadURL;

	@Value("${image.dpi}")
	private int imageDpi;

	@Value("${temp.rectangle.folder}")
	private String rectangleFolder;

	@Value("${secret.key}")
	private String secret;

	@Value("${min_in_config}")
	private int minMinutes;

	public Map<String, Object> verifyAndPdfDetails(Authorization auth) throws Exception {

		// Get Token and Authorize it first
		File pdfFile = null;
		Map<String, Object> dataMap = null;
		try {
			if (isRequestTokenValid(auth)) {
				dataMap = new HashMap<String, Object>();
				pdfFile = downloadPDFFile(auth.getDocumentId(), auth.getTimestamp(), auth.getToken());
				dataMap.put("filename", pdfFile.getName());
				dataMap.put("totalPages", getTotalPagesInPDF(pdfFile));
				dataMap.put("dpi", imageDpi);
				// getHeightWidthAndCordinates(pdfFile);
				pdfFile = new File(tempDirectory + File.separator + rectangleFolder);
				if (pdfFile.exists() && pdfFile.isDirectory()) {
					try (Stream<Path> walk = Files.walk(Paths.get(pdfFile.getAbsolutePath()))) {
						List<String> result = walk.filter(Files::isRegularFile).map(x -> x.getFileName().toString())
								.collect(Collectors.toList());
						dataMap.put("rectangles", result);

					} catch (IOException e) {
						e.printStackTrace();
					}
				}

			} else {
				throw new Exception("Not Authorized");
			}
		} catch (Exception e) {
			throw e;
		}
		return dataMap;
	}

	public File getImage(String pdfFileName, int pageNumber) throws Exception {
		File pdfFile = null;
		File image = null;
		try {
			pdfFile = new File(new File(tempDirectory), pdfFileName);
			if (pdfFile.exists() && pdfFile.isFile()) {
				image = convertPdfToImage(pdfFile, pageNumber);
			} else {
				throw new Exception("File does not exist: " + pdfFile.getName());
			}
		} catch (Exception e) {
			throw e;
		}
		return image;
	}

	private File downloadPDFFile(String docId, long timestamp, String token) {
		RestTemplate restTemplate = new RestTemplate();
		String url = pdfDownloadURL + "?documentId=" + docId + "&timestamp=" + timestamp + "&token=" + token;
		//String url = "http://localhost:8080/getByteArray";
		File file = restTemplate.execute(url, HttpMethod.GET, null, clientHttpResponse -> {
			File tempFile = File.createTempFile(docId, ".pdf", new File(tempDirectory));
			try (InputStream is = clientHttpResponse.getBody(); OutputStream os = new FileOutputStream(tempFile)) {
				StreamUtils.copy(is, os);
			}
			return tempFile;
		});
		return file;
	}

	private File convertPdfToImage(File pdfFile, int pageNumber) throws Exception {
		File imgFile = new File(new File(tempDirectory),
				FilenameUtils.removeExtension(pdfFile.getName()) + "-" + pageNumber + ".png");
		try (FileInputStream is = new FileInputStream(pdfFile);
				PDDocument document = PDDocument.load(is);
				OutputStream os = new FileOutputStream(imgFile)) {
			PDFRenderer pdfRenderer = new PDFRenderer(document);
			if (pageNumber >= document.getNumberOfPages()) {
				throw new Exception("Provided Page Number is out of range. Pdf has only " + document.getNumberOfPages()
						+ " pages.");
			}
			BufferedImage bim = pdfRenderer.renderImageWithDPI(pageNumber, imageDpi, ImageType.RGB);
			ImageIOUtil.writeImage(bim, "png", os, imageDpi, 1.0f);
		} catch (Exception e) {
			throw e;
		}
		return imgFile;
	}

	private int getTotalPagesInPDF(File pdfFile) throws Exception {
		int totalPages = 0;
		try (FileInputStream is = new FileInputStream(pdfFile); PDDocument document = PDDocument.load(is);) {
			totalPages = document.getNumberOfPages();
		} catch (Exception e) {
			throw e;
		}
		return totalPages;
	}

	public void cleanUp(String pdfFileName, int totalPages) throws Exception {
		boolean flag = true;
		List<String> fileList = new ArrayList<>();
		if (pdfFileName != null && totalPages >= 0) {
			File file = null;
			try {
				file = new File(new File(tempDirectory), pdfFileName);
				if (file.exists() && file.isFile()) {
					flag = file.delete();
					if (!flag) {
						fileList.add(file.getName());
					}
				}
				for (int i = 0; i < totalPages; i++) {
					file = new File(new File(tempDirectory),
							FilenameUtils.removeExtension(pdfFileName) + "-" + i + ".png");
					if (file.exists() && file.isFile()) {
						flag = file.delete();
						if (!flag) {
							fileList.add(file.getName());
						}
					}
				}
			} catch (Exception e) {
				throw e;
			}

		}
		if (fileList.size() > 0) {
			throw new Exception("File Delete error. Files which are not deleted are: "
					+ fileList.stream().collect(Collectors.joining(",")));
		}
	}

	// For testing purpose only
	public byte[] getByteArray() throws FileNotFoundException {
		String path = "C:\\Users\\Shahid Sabadia\\Desktop\\ABC.pdf";
		File file = new File(path);
		byte[] bytesArray = new byte[(int) file.length()];

		try (FileInputStream fis = new FileInputStream(file)) {
			fis.read(bytesArray);
		} catch (IOException e) {
			e.printStackTrace();
		}

		return bytesArray;
	}

	/*
	 * private int getHeightWidthAndCordinates(File pdfFile) throws Exception { int
	 * totalPages = 0; try (FileInputStream is = new FileInputStream(pdfFile);
	 * PDDocument document = PDDocument.load(is);) { // PDPage page =
	 * document.getPage(0); width = page.getMediaBox().getWidth(); height =
	 * page.getMediaBox().getHeight(); System.out.println("width: " + width);
	 * System.out.println("height: " + height);
	 * 
	 * HelperClass hc = new HelperClass(); hc.process(document); // totalPages =
	 * document.getNumberOfPages(); } catch (Exception e) { throw e; } return
	 * totalPages; }
	 */

	private boolean isRequestTokenValid(Authorization auth) throws Exception {
		String token1 = null;
		boolean validToken = false;
		Calendar now = null;
		Calendar cal = null;
		String timeString = null;
		try {
			token1 = getToken(auth);
			if (token1.equals(auth.getToken())) {
				timeString = String.valueOf(auth.getTimestamp());
				cal = Calendar.getInstance();
				cal.clear();
				cal.set(Calendar.DATE, Integer.valueOf(timeString.substring(6, 8)));
				cal.set(Calendar.MONTH, Integer.valueOf(timeString.substring(4, 6)) - 1);
				cal.set(Calendar.YEAR, Integer.valueOf(timeString.substring(0, 4)));
				cal.set(Calendar.HOUR_OF_DAY, Integer.valueOf(timeString.substring(8, 10)));
				cal.set(Calendar.MINUTE, Integer.valueOf(timeString.substring(10)));
				cal.add(Calendar.MINUTE, minMinutes);
				now = Calendar.getInstance();
				if (now.before(cal)) {
					validToken = true;
				}
			}
		} catch (Exception e) {
			throw e;
		}
		return validToken;
	}

	private String getToken(Authorization auth) throws Exception {
		String hash = "";
		Mac sha256_HMAC = null;
		SecretKeySpec secret_key = null;
		String message = null;
		try {
			message = auth.getDocumentId() + auth.getTimestamp();
			sha256_HMAC = Mac.getInstance("HmacSHA256");
			secret_key = new SecretKeySpec(secret.getBytes("UTF-8"), "HmacSHA256");
			sha256_HMAC.init(secret_key);
			hash = new String(Hex.encodeHex(sha256_HMAC.doFinal(message.getBytes("UTF-8"))));
			// hash =
			// Base64.encodeBase64String(sha256_HMAC.doFinal(message.getBytes("UTF-8")));
		} catch (Exception e) {
			throw e;
		}
		return hash;
	}

}
