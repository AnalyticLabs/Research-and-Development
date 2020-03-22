package com.healthnavigatorapis.portal.chatbot.data.remote.model;

import com.google.gson.annotations.SerializedName;

public class QuestionPrimary {

    @SerializedName("ElementPLQuestion")
    private String elementPLQuestion;

    @SerializedName("ResultStatus")
    private int resultStatus;

    @SerializedName("ResultStatusDescription")
    private String resultStatusDescription;

    @SerializedName("Title")
    private String title;

    @SerializedName("EvaluationAcuityRatingColorImgURL")
    private String evaluationAcuityRatingColorImgURL;

    @SerializedName("DocumentationElementOrder")
    private int documentationElementOrder;

    @SerializedName("ConceptId")
    private int conceptId;

    @SerializedName("DocumentationType")
    private String documentationType;

    @SerializedName("DocumentationTypeOrder")
    private int documentationTypeOrder;

    @SerializedName("EvaluationAcuityRatingNumeric")
    private int evaluationAcuityRatingNumeric;

    @SerializedName("ConceptOrder")
    private int conceptOrder;

    @SerializedName("TitlePlainLanguage")
    private String titlePlainLanguage;

    @SerializedName("DocumentationElement")
    private String documentationElement;

    @SerializedName("ElementPLQuestionLocalized")
    private String elementPLQuestionLocalized;

    @SerializedName("_inputConceptId")
    private int inputConceptId;

    @SerializedName("Question")
    private String question;

    @SerializedName("ParentConceptId")
    private int parentConceptId;

    @SerializedName("FrequencyRatingNumeric")
    private int frequencyRatingNumeric;

    @SerializedName("TitleLocalized")
    private String titleLocalized;

    public String getElementPLQuestion() {
        return elementPLQuestion;
    }

    public void setElementPLQuestion(String elementPLQuestion) {
        this.elementPLQuestion = elementPLQuestion;
    }

    public int getResultStatus() {
        return resultStatus;
    }

    public void setResultStatus(int resultStatus) {
        this.resultStatus = resultStatus;
    }

    public String getResultStatusDescription() {
        return resultStatusDescription;
    }

    public void setResultStatusDescription(String resultStatusDescription) {
        this.resultStatusDescription = resultStatusDescription;
    }

    public String getTitle() {
        return title;
    }

    public void setTitle(String title) {
        this.title = title;
    }

    public String getEvaluationAcuityRatingColorImgURL() {
        return evaluationAcuityRatingColorImgURL;
    }

    public void setEvaluationAcuityRatingColorImgURL(String evaluationAcuityRatingColorImgURL) {
        this.evaluationAcuityRatingColorImgURL = evaluationAcuityRatingColorImgURL;
    }

    public int getDocumentationElementOrder() {
        return documentationElementOrder;
    }

    public void setDocumentationElementOrder(int documentationElementOrder) {
        this.documentationElementOrder = documentationElementOrder;
    }

    public int getConceptId() {
        return conceptId;
    }

    public void setConceptId(int conceptId) {
        this.conceptId = conceptId;
    }

    public String getDocumentationType() {
        return documentationType;
    }

    public void setDocumentationType(String documentationType) {
        this.documentationType = documentationType;
    }

    public int getDocumentationTypeOrder() {
        return documentationTypeOrder;
    }

    public void setDocumentationTypeOrder(int documentationTypeOrder) {
        this.documentationTypeOrder = documentationTypeOrder;
    }

    public int getEvaluationAcuityRatingNumeric() {
        return evaluationAcuityRatingNumeric;
    }

    public void setEvaluationAcuityRatingNumeric(int evaluationAcuityRatingNumeric) {
        this.evaluationAcuityRatingNumeric = evaluationAcuityRatingNumeric;
    }

    public int getConceptOrder() {
        return conceptOrder;
    }

    public void setConceptOrder(int conceptOrder) {
        this.conceptOrder = conceptOrder;
    }

    public String getTitlePlainLanguage() {
        return titlePlainLanguage;
    }

    public void setTitlePlainLanguage(String titlePlainLanguage) {
        this.titlePlainLanguage = titlePlainLanguage;
    }

    public String getDocumentationElement() {
        return documentationElement;
    }

    public void setDocumentationElement(String documentationElement) {
        this.documentationElement = documentationElement;
    }

    public String getElementPLQuestionLocalized() {
        return elementPLQuestionLocalized;
    }

    public void setElementPLQuestionLocalized(String elementPLQuestionLocalized) {
        this.elementPLQuestionLocalized = elementPLQuestionLocalized;
    }

    public int getInputConceptId() {
        return inputConceptId;
    }

    public void setInputConceptId(int inputConceptId) {
        this.inputConceptId = inputConceptId;
    }

    public String getQuestion() {
        return question;
    }

    public void setQuestion(String question) {
        this.question = question;
    }

    public int getParentConceptId() {
        return parentConceptId;
    }

    public void setParentConceptId(int parentConceptId) {
        this.parentConceptId = parentConceptId;
    }

    public int getFrequencyRatingNumeric() {
        return frequencyRatingNumeric;
    }

    public void setFrequencyRatingNumeric(int frequencyRatingNumeric) {
        this.frequencyRatingNumeric = frequencyRatingNumeric;
    }

    public String getTitleLocalized() {
        return titleLocalized;
    }

    public void setTitleLocalized(String titleLocalized) {
        this.titleLocalized = titleLocalized;
    }
}