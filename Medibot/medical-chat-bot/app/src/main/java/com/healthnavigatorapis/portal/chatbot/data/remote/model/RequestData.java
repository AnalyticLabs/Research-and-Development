package com.healthnavigatorapis.portal.chatbot.data.remote.model;

import android.text.TextUtils;

import java.util.ArrayList;
import java.util.Calendar;
import java.util.Date;
import java.util.List;

public class RequestData {
    private int conceptId;
    private int ageInDays;
    private String gender;
    private ArrayList<Integer> secondaryCCCPresent = new ArrayList<>();
    private ArrayList<Integer> conceptPresent = new ArrayList<>();
    private String primaryComplaint;
    private List<String> secondaryComplaint = new ArrayList<>();

    public int getConceptId() {
        return conceptId;
    }

    public void setConceptId(String value, int conceptId) {
        this.conceptId = conceptId;
        primaryComplaint = value;
        removeConceptFromPresent(value);
    }

    public int getAgeInDays() {
        return ageInDays;
    }

    public void setAgeInDays(int age) {
        Calendar cal = Calendar.getInstance();
        Date today = cal.getTime();
        cal.set(cal.get(Calendar.YEAR) - age, Calendar.JANUARY, 1);
        Date birthday = cal.getTime();

        long dateSubtract = today.getTime() - birthday.getTime();
        long time = 1000 * 60 * 60 * 24;
        ageInDays = (int) (dateSubtract / time);
    }

    public void setAge(int age) {
        ageInDays = age;
    }

    public String getGender() {
        if (!TextUtils.isEmpty(gender)) {
            return String.valueOf(gender.charAt(0)).toUpperCase();
        } else {
            return String.valueOf("B");
        }
    }

    public void setGender(String gender) {
        this.gender = gender;
    }

    public Integer[] getConceptPresent() {
        return conceptPresent.toArray(new Integer[0]);
    }

    public void addConceptPresent(int conceptId) {
        conceptPresent.add(conceptId);
    }

    public Integer[] getSecondaryCCCPresent() {
        if (secondaryCCCPresent.isEmpty()) {
            return null;
        }
        return secondaryCCCPresent.toArray(new Integer[0]);
    }

    public void addSecondaryCCCPresent(String string, int conceptId) {
        secondaryCCCPresent.add(conceptId);
        secondaryComplaint.add(string);
    }

    private void removeConceptFromPresent(String value) {
        conceptPresent.remove(Integer.valueOf(conceptId));
        secondaryCCCPresent.remove(Integer.valueOf(conceptId));
        secondaryComplaint.remove(value);
    }

    public void clear() {
        secondaryCCCPresent.clear();
        conceptPresent.clear();
        secondaryCCCPresent.clear();
        primaryComplaint = null;
    }

    public String getPrimaryComplaint() {
        return primaryComplaint;
    }

    public void setPrimaryComplaint(String primaryComplaint) {
        this.primaryComplaint = primaryComplaint;
    }

    public List<String> getSecondaryComplaint() {
        return secondaryComplaint;
    }

    public void setSecondaryComplaint(List<String> secondaryComplaint) {
        this.secondaryComplaint = secondaryComplaint;
    }
}
