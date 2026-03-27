用于改变仿真器的Hook

实现如下的功能： 做为Hook在某个指定时间点被调用

class Hook:
    def invoke(self, 
    invoke_type(以下6种之一 ：pre_action_step , post_action_step , pre_phy_step , post_phy_step , pre_episode , post_episode) , 
    f_get_action, 
    f_get_static_data ,  
    f_get_sensor_data , 
    f_get_core_state , 
    f_get_derived_state , 
    f_set_core_state (如果不想让修改， 给None), 
    f_set_action (如果不想让修改， 给None)): -> bool (teminate flag)
       