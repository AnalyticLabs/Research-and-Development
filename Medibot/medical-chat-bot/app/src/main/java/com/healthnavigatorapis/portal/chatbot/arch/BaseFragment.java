package com.healthnavigatorapis.portal.chatbot.arch;

import android.content.Context;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;

import com.healthnavigatorapis.portal.chatbot.ui.main.FragmentNavigator;

import androidx.annotation.LayoutRes;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.databinding.DataBindingUtil;
import androidx.databinding.ViewDataBinding;
import androidx.fragment.app.Fragment;

public abstract class BaseFragment<T extends ViewDataBinding> extends Fragment {

    private T mViewDataBinding;
    private FragmentNavigator mFragmentNavigator;

    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle savedInstanceState) {
        mViewDataBinding = DataBindingUtil.inflate(inflater, getLayoutId(), container, false);
        return mViewDataBinding.getRoot();
    }

    @LayoutRes
    protected abstract int getLayoutId();

    public T getBinding() {
        return mViewDataBinding;
    }

    @Override
    public void onAttach(Context context) {
        super.onAttach(context);
        try {
            mFragmentNavigator = ((FragmentNavigator) context);
        } catch (ClassCastException e) {
            throw new ClassCastException(context.toString() + " must implement MainNavigator");
        }
    }

    @Override
    public void onDetach() {
        super.onDetach();
        mFragmentNavigator = null;
    }

    public FragmentNavigator getNavigator() {
        return mFragmentNavigator;
    }
}