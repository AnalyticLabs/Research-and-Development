package com.ss.conversion.model;

import java.util.List;

public class Authorization {
	private List<Signer> signers;
	private String documentId;
	private long timestamp;
	private String token;

	public List<Signer> getSigners() {
		return signers;
	}

	public void setSigners(List<Signer> signers) {
		this.signers = signers;
	}

	public String getDocumentId() {
		return documentId;
	}

	public void setDocumentId(String documentId) {
		this.documentId = documentId;
	}

	public long getTimestamp() {
		return timestamp;
	}

	public void setTimestamp(long timestamp) {
		this.timestamp = timestamp;
	}

	public String getToken() {
		return token;
	}

	public void setToken(String token) {
		this.token = token;
	}

}
