package com.healthnavigatorapis.portal.chatbot;

public final class Constants {
    public static final String BASE_URL = "https://sandbox.healthnavigatorapis.com/3.0/";
    public static final String FIND_CCC = "FindCcc";
    public static final String TELL_US_MORE_PRIMARY_CCC_OPQRST = "TellUsMorePrimaryCCC_OPQRST";
    public static final String TELL_US_MORE_SECONDARY_CCC_OPQRST = "TellUsMoreSecondaryCCC_OPQRST_Severity";
    public static final String TELL_US_MORE_PRIMARY_CCC = "TellUsMorePrimaryCCC";
    public static final String GET_CAUSES = "GetCauses";
    public static final String CALCULATE_TRIAGE_SCORE = "CalculateTriageScore";

    public static final String MESSAGE_NOT_CHOOSE = "Nope";

    public static final int BOT_ID = 0;
    public static final int USER_ID = 1;
    public static final int SOMEONE_ID = 2;

    public static final int MAX_CAUSES_SIZE = 20;
    public static final int DELAY_MILLI = 500;

    public static final int RESULT_SPEECH_RECOGNITION = 1;

    private Constants() {
    }
}
