package com.healthnavigatorapis.portal.chatbot.ui.chat;

import android.view.LayoutInflater;
import android.view.ViewGroup;

import com.healthnavigatorapis.portal.chatbot.BR;
import com.healthnavigatorapis.portal.chatbot.R;
import com.healthnavigatorapis.portal.chatbot.data.local.model.Message;
import com.healthnavigatorapis.portal.chatbot.databinding.ItemLeftChatBinding;
import com.healthnavigatorapis.portal.chatbot.interfaces.IChoicesPressed;

import java.util.ArrayList;

import androidx.annotation.NonNull;
import androidx.databinding.DataBindingUtil;
import androidx.databinding.ViewDataBinding;
import androidx.recyclerview.widget.RecyclerView;

public class ChatAdapter extends RecyclerView.Adapter<ChatAdapter.ViewHolder> {

    private static final int LEFT_TYPE = 0;
    private static final int RIGHT_TYPE = 1;
    private static final int CENTER_TYPE = 2;

    private ArrayList<Message> mMessages = new ArrayList<>();
    private IChoicesPressed mListener;

    @NonNull
    @Override
    public ViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        if (viewType == LEFT_TYPE) {
            return new ViewHolder(DataBindingUtil.inflate(LayoutInflater.from(parent.getContext()),
                    R.layout.item_left_chat, parent, false));
        } else if (viewType == RIGHT_TYPE) {
            return new ViewHolder(DataBindingUtil.inflate(LayoutInflater.from(parent.getContext()),
                    R.layout.item_right_chat, parent, false));
        } else {
            return new ViewHolder(DataBindingUtil.inflate(LayoutInflater.from(parent.getContext()),
                    R.layout.item_triage, parent, false));
        }
    }

    public void addMessage(Message message) {
        mMessages.add(message);
        notifyItemInserted(mMessages.size() - 1);
    }

    public void clearMessages() {
        int size = mMessages.size();
        mMessages.clear();
        notifyItemRangeRemoved(0, size);
    }

    @Override
    public void onBindViewHolder(@NonNull ViewHolder holder, int position) {
        holder.bind(mMessages.get(position));
    }

    @Override
    public int getItemCount() {
        return mMessages == null ? 0 : mMessages.size();
    }

    @Override
    public int getItemViewType(int position) {
        switch (mMessages.get(position).getPosition()) {
            case RIGHT:
                return RIGHT_TYPE;
            case LEFT:
                return LEFT_TYPE;
            case CENTER:
                return CENTER_TYPE;
        }
        return CENTER_TYPE;
    }

    public void setListener(IChoicesPressed listener) {
        mListener = listener;
    }

    public class ViewHolder extends RecyclerView.ViewHolder {
        ViewDataBinding binding;

        public ViewHolder(@NonNull ViewDataBinding itemView) {
            super(itemView.getRoot());
            binding = itemView;
        }

        public void bind(Message message) {
            binding.setVariable(BR.message, message);
            if (message.getPosition() == Message.Position.LEFT) {
                if (mListener != null) {
                    ((ItemLeftChatBinding) binding).leftChatChoices.setListener(mListener);
                }
            }
            binding.executePendingBindings();
        }
    }
}
