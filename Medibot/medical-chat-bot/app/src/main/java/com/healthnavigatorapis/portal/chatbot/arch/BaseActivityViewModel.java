package com.healthnavigatorapis.portal.chatbot.arch;

import android.os.Bundle;

import androidx.annotation.Nullable;
import androidx.databinding.ViewDataBinding;
import androidx.lifecycle.ViewModel;

public abstract class BaseActivityViewModel<T extends ViewDataBinding, V extends ViewModel> extends BaseActivity<T> {

    private V mViewModel;

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        mViewModel = initViewModel();
        getBinding().setVariable(getBindingVariable(), mViewModel);
        getBinding().executePendingBindings();
    }

    public V getViewModel() {
        return mViewModel;
    }

    public abstract V initViewModel();

    public abstract int getBindingVariable();
}
