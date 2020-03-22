package com.healthnavigatorapis.portal.chatbot.arch;

import android.os.Bundle;
import android.view.View;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.databinding.ViewDataBinding;
import androidx.lifecycle.ViewModel;

public abstract class BaseFragmentViewModel<T extends ViewDataBinding, V extends ViewModel> extends BaseFragment<T> {

    private V mViewModel;


    @Override
    public void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        mViewModel = initViewModel();
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);
        getBinding().setVariable(getBindingVariable(), mViewModel);
        getBinding().executePendingBindings();
        setSubscribers();
    }

    public abstract V initViewModel();

    public V getViewModel() {
        return mViewModel;
    }

    public abstract int getBindingVariable();

    protected abstract void setSubscribers();
}